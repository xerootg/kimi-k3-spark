#!/usr/bin/env bash
# Check the committed upstream tree; never modify the supplied checkout.
set -euo pipefail
export LC_ALL=C

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# != 1 ]]; then
    echo "Usage: $0 /path/to/llama.cpp (HEAD must match versions.env LLAMA_SHA)" >&2
    exit 2
fi
upstream=$(git -C "$1" rev-parse --show-toplevel)
sha=$(sed -n 's/^LLAMA_SHA=//p' "$repo/versions.env")
if [[ ! $sha =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: versions.env must contain one full LLAMA_SHA" >&2
    exit 1
fi
if [[ $(git -C "$upstream" rev-parse HEAD) != "$sha" ]]; then
    echo "ERROR: $upstream HEAD must equal LLAMA_SHA=$sha" >&2
    exit 1
fi

work=$(mktemp -d "$repo/.patch-stack-check.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
mkdir "$work/tree"
git -C "$upstream" archive HEAD | tar -x -C "$work/tree"
git -C "$work/tree" init -q
shopt -s nullglob
patches=("$repo"/patches/*.patch)
if (( ${#patches[@]} == 0 )); then
    echo "ERROR: no patches found" >&2
    exit 1
fi
for patch in "${patches[@]}"; do
    echo "Checking and applying ${patch##*/}"
    if ! git -C "$work/tree" apply --check "$patch"; then
        echo "ERROR: patch rejected: ${patch##*/}" >&2
        exit 1
    fi
    if ! git -C "$work/tree" apply "$patch"; then
        echo "ERROR: applying patch failed: ${patch##*/}" >&2
        exit 1
    fi
    git -C "$work/tree" apply --numstat "$patch" >> "$work/touched"
done

# Include nested src/models/*.cpp as well as the top-level src files.
# Headers are checked through their translation units; CUDA and C are excluded.
awk -F '\t' '$3 ~ /^(src\/|ggml\/src\/ggml-cpu\/).*\.cpp$/ { print $3 }' \
    "$work/touched" | sort -u > "$work/cpp"
cd "$work/tree"
check_cpp() {
    local file=$1
    # A later patch may have deleted a translation unit.
    [[ -f $file ]] || return 0
    echo "Syntax checking $file"
    if ! "${CXX:-c++}" -std=c++17 -fsyntax-only -pthread -D_GNU_SOURCE \
        -Iinclude -Isrc -Isrc/models -Iggml/include -Iggml/src \
        -Iggml/src/ggml-cpu "$file"; then
        echo "ERROR: syntax check failed: $file" >&2
        return 1
    fi
}
export -f check_cpp
# Bound concurrency and memory use on the two-core GitHub runner.
xargs -r -d '\n' -P 2 -I '{}' bash -c 'check_cpp "$1"' _ '{}' < "$work/cpp"
echo "PASS: ${#patches[@]} patches applied; $(wc -l < "$work/cpp") C++ files checked"

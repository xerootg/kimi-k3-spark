#!/usr/bin/env python3
"""Offline validation on a local copy of the supplied applied source, never on GH200."""
from pathlib import Path
import difflib
import subprocess
import shutil
root = Path(__file__).resolve().parents[2]
tree = root / '.stride-check'
reference = root.parent/'sol-hooks/checks/0020c-pre'
if not (tree/'ggml').is_dir():
    shutil.copytree(reference, tree, ignore=shutil.ignore_patterns('.git'), dirs_exist_ok=True)
for rel in ['src/llama-moe-cache.cpp', 'src/llama-moe-cache.h']:
    shutil.copyfile(reference/rel, tree/rel)
p = tree / 'src/llama-moe-cache.cpp'
s = p.read_text()
# Bring the supplied older 0004 guard to this repository's 0004 postimage.
patch = (root / 'patches/0004-byte-stride-slots.patch').read_text()
hunk = patch.split('@@ -2793,13 +2794,39 @@')[1].split('\n@@')[0]
post = ''.join(line[1:]+'\n' for line in hunk.splitlines()[1:] if line.startswith(('+',' ')))
a = s.index('                // Opt in until CUDA compilation')
b = s.index('                for (int32_t il : g.layers) {', s.index('g.slot_stride += 4096;', a))
# Extract exactly the guard through the stride log, leaving matching layout check intact.
start = post.index('                // Opt in until CUDA compilation')
end = post.index('                for (int32_t il : g.layers) {', post.index('g.slot_stride += 4096;'))
s = s[:a] + post[start:end] + s[b:]
p.write_text(s)
h = tree / 'src/llama-moe-cache.h'
hs = h.read_text()
if 'inline size_t llama_moe_cache_cuda_alignment' not in hs:
    helpers = patch.split('+// Minimum bank alignment')[1].split('\n struct llama_model;')[0]
    helpers = '// Minimum bank alignment' + helpers.replace('\n+', '\n')
    hs = hs.replace('struct llama_model;', helpers + '\nstruct llama_model;')
    h.write_text(hs)
old = s
s = s.replace('                const char * env_slots = getenv("LLAMA_EXPERT_STORE_GPU_SLOTS");\n', '')
s = s.replace(" && !(env_slots != nullptr && env_slots[0] == '0')", '')
s = s.replace('needs CUDA GPU slots and host-bank hooks', 'needs CUDA host-bank hooks')
block = '''            if (!slots_on_gpu && std::any_of(groups.begin(), groups.end(), [](const group & g) { return g.byte_stride; })) {
                throw std::runtime_error("expert store: byte stride requires GPU slot maps; retry with LLAMA_EXPERT_STORE_BYTE_STRIDE=0");
            }
'''
assert block in s
s = s.replace(block, '''            // CPU GET_ROWS reads I32 slot IDs only; CUDA still reads registered
            // banks using byte strides. Host-bank registration failure stays fatal.
''')
# Small contexts apply after 0004, independently of later hook additions.
diff = 'diff --git a/src/llama-moe-cache.cpp b/src/llama-moe-cache.cpp\n' + ''.join(difflib.unified_diff(old.splitlines(True), s.splitlines(True), fromfile='a/src/llama-moe-cache.cpp', tofile='b/src/llama-moe-cache.cpp', n=3))
(root/'patches/0004b-byte-stride-k3-types.patch').write_text(diff)
subprocess.run(['patch', '--dry-run', '-p1', '-i', str(root/'patches/0004b-byte-stride-k3-types.patch')], cwd=tree, check=True)
subprocess.run(['patch', '-p1', '-i', str(root/'patches/0004b-byte-stride-k3-types.patch')], cwd=tree, check=True)
subprocess.run(['c++','-std=c++17','-fsyntax-only','-pthread','-D_GNU_SOURCE','-Iinclude','-Isrc','-Isrc/models','-Iggml/include','-Iggml/src','-Iggml/src/ggml-cpu','src/llama-moe-cache.cpp'],cwd=tree,check=True)
print('PASS: 0004b applies to local reference with current 0004 guard; cache C++ -fsyntax-only')

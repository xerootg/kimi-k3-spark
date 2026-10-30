#!/bin/bash
# asmstat.sh DUMPDIR REGEX: per IGC .asm whose .kernel name matches REGEX (case-insensitive): instruction count,
# dp4a, integer mul/mach, sends (memory), and the demangled-ish kernel name.
cd "$1"
for f in *.asm; do
  k=$(command grep -a -m1 '^//.kernel' "$f" | cut -c10-)
  echo "$k" | command grep -qiE "$2" || continue
  tot=$(command grep -a -cE '^\s*(\(W[^)]*\)\s+)?[a-z][a-z0-9.]* \(' "$f")
  dp=$(command grep -a -cE '\sdp4a ' "$f"); mul=$(command grep -a -cE '\s(mul|mach|mad) \(' "$f"); snd=$(command grep -a -cE '\ssends?c? \(|\sload|\sstore' "$f")
  short=$(echo "$k" | sed -E 's/_ZTSZ+L?[0-9]*//; s/ILi.*//; s/(Ev|EN).*//' | cut -c1-70)
  printf "%-70s instr %5d dp4a %4d mul/mad %4d send %4d %s\n" "$short" $tot $dp $mul $snd "$f"
done | sort | uniq -w 70

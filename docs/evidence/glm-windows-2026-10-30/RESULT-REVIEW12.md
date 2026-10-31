Verdict: fix one P1 dispatch omission before acceptance; source math/address audit passes, native identity and performance remain unverified.
Scope: inputs/c-tree untouched; cp -a copies candidate=(0027 + alignment + 0028), alone=(0028), fixed=(alone + reviewer fix). No network/commit/push/rm.
References: M=review12/candidate/ggml/src/ggml-cuda/mmvq.cu; B=bank-gemv.cuh there; C/U=src/llama-moe-cache.cpp / ggml-cuda.cu in c-tree.
P1 M:1820-1824: case-1 lambda calls switch_fusion(..., stream) without is_bank; default false disables bank_schedule. Other column cases forward it but fail c_ncols_dst==1.
Therefore every intended decode bank kernel stays stock, independent of 0027; bank_gemv_enabled() is never reached and no bank init log appears. Standalone bench bypasses this dispatch.
Fix: review12/fixes/0001-forward-bank-provenance.patch forwards is_bank in that lambda. git apply --check PASS after 0028 in both stacks; applied/reverse-checked in fixed copy.
Read 1, per-lane order: tid=32*warp+lane; kqs=vdr*(tid%8), start=tid/8, stride=16 with NW4. Flatten (base,g) gives start+n*16 in exactly stock order.
IQ2/3 K6144: starts 0..7 consume two blocks, 8..15 one; K2048 consumes one/zero. IQ3 small_k explicitly disabled; IQ4 K2048 uses small_k, four rows and one/zero blocks.
Every tmp[0][i] and tmp_gate[0][i] receives the same helper returns in ascending kbx; unchanged +0 initialization, original activation kby=kbx*8, local helper block index zero.
Read 2, fields: IQ2 d + qs bytes[4*kqs,4*kqs+8) + scales[kqs/2]; IQ3 d + qs[4*kqs,+8) and qs[64+2*kqs,+4); IQ4 d/scales_h/scales_l[kqs/8] + qs[4*kqs,+16).
Executed actual bank_fetch helpers with portable vector loads: 6144 block/lane cases preserve every consumed byte; supplied IQ dots match inherited review9 vecdotq exactly (target vecdotq absent).
IQ4 four-row staging performs only byte/half-field transfers and integer addressing before consumption; no FP conversion, multiply, dot or addition is moved across another operation on the same accumulator.
Reduction/fusion suffix is byte-for-byte stock: warps 1..3 added ascending, XOR warp sum, same bias/GLU/fused gate-up/output positions. Interleaving independent rows does not reassociate their sums.
M original :681 finite-scale dummy branch is byte-identical and precedes bank loads; nonfinite scales still run original zero-record dot/reduction/fusion. Grid/shared-last expert handling unchanged.
Source equivalence does not certify generated FP contraction/FTZ/NaN payloads after specialization; need native off/on bytes and generated instructions with actual target flags (bench omits fast-math).
Addressing: unchanged int64 sample/channel byte offset, s02=nb[2], s03=nb[3], then original row/block indices. No slot stride converted to blocks or narrowed to 32 bits.
C:3151-3190 places gate/up/down at region+part_off, same slot_stride, inherited HostBank views; n_slots+1 storage includes cleared dummy slot C:3344. CPU/device maps and merges untouched.
GPU GET_ROWS still produces ids[channel_dst]; CPU-filled maps still feed the existing transfer/lookup route. 0028 never rewrites map entries, gathers, expert IDs or dummy positions.
Overfetch: B rounds each consumed span outward to 16B; aligned row start/end guarantee every rounded interval stays inside that row, though it may include neighboring quant blocks.
Exhaustive 233472 span checks cover all lanes and aligned row lengths through 256 blocks. Thus no next-slot/bank-end read, even last row of dummy; extra valid mapped bytes cost bandwidth, no new boundary fault.
Register estimate, added useful staged scalars at three groups: IQ2 ~12 regs unfused/~24 fused; IQ3 ~12/~24; IQ4 four-row ~84/~168 if all three groups live (~28/~56 for one valid group).
These exclude dots, pointers, extraction temporaries, accumulators and stock prologue; IQ4 down is unfused. Runtime K means one-trip shapes do not guarantee compiler elimination of other groups.
Declared bx storage/thread: IQ2 222B (444B fused), IQ3 294B (588B fused), IQ4 R4 1632B (3264B fused), plus bg even without fusion if not eliminated; these are arrays, not measured spills.
Runtime kqs-indexed local structs/byte extraction may prevent scalar replacement. Require ptxas local/spill counts and SASS/local traffic for production kernels; runtime kill switch cannot undo shared kernel resource allocation.
After fix guard: HostBank name + IDs + cc121/48SM + host_bank capability + compiled121 + 16B bases/row/slot/sample alignment. CPU buffers and device src0 reject; generic fallback entry defaults is_bank=false.
Shared expert CTA stays stock; multi-token ID uses dedicated old MoE kernel. Fused gate gets alignment check but no separate buffer provenance test; supplied constructor gives gate/up/down the same HostBank region.
Kill switch: getenv cached once; unset enables, atoi==0 disables (including literal 0, empty/non-numeric). No new allocations/copies/waits; existing activation quantization unchanged.
0020 wait IS present in U:1817: cuStreamWaitValue32 before lookup and downstream GEMV; poller fences hook writes before done U:1709-1712. Both files unchanged by 0028.
Bench: mapped cudaHostRegister weights, copies=ceil(512MiB/live payload), iters>=copies; finite activation timing skips dummy rows, rotates >=512MiB useful unfused weights, twice that fused; no CPU writes in timed loop.
CHECK compares bytewise to reproduced stock over random finite weights, zero activation scales, NaN/Inf activation scales and fixed dummy suffixes. Both paths read same seed/copy; dummy zero record initialized.
Coverage limits: no real dispatch/maps, nonprefix dummy permutations, >32-bit slot offsets, shared expert, arbitrary GLUs/biases or native target flags; CHECK success is not production certification. No bank speed measured here.
Window bash -n PASS: native rebuild, standalone SUMMARY gate, BG0/BG1/BG0b/BG1b, isolated snapshots, health, three replies/leg, predicted_n=128 + gold MD5, hooks/dense/bank log presence. Unfixed P1 aborts bank-log gate.
P2 window: init presence does not prove actual launch/mode; no finite-positive timing, spill, sanitizer, NaN/error-log, speedup/repeatability or clean-exit gate; stop_server discards exit status. Dense remains on for both modes.
Checks: python3 review12/check.py > review12/logs/check.log 2>&1 -> SUMMARY: one P1 dispatch omission; fix apply-check PASS in both stacks; offline checks PASS. No nvcc/GPU/model/window run.

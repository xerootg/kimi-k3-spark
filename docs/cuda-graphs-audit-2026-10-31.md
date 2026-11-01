The available source does not confirm an exact-mode bypass during capture.
This audit uses evidence/c-tree and evidence/c-tree-extra plus the installed
patch artifacts; the actual ~/llama-k3-fd6 tree is absent from this sandbox.

| Path | Capture/replay behavior visible in source |
| --- | --- |
| 0030 v2 split maps and graph construction | Eligibility depends on token count, selected experts, remote mode and flags. No capture predicate. MMVQ bank splitting launches one column at a time on the supplied stream. |
| 0008 MMVQ exact columns | Recurses into the singleton dispatcher, with per-column IDs, output and shared-destination offsets. No capture predicate or host sync added. |
| 0035 v4 ordinary and fused matvec | Per-column recursion, pool scratch, launches and device-to-device async scatter. No capture predicate or explicit host sync added. Nested backend operations and scratch lifetime still need CUDA validation. |
| 0036 v2 attention/TOP_K | Slicing happens while constructing the GGML graph, before CUDA capture. Per-sequence nodes and concatenation remain graph nodes. There is no capture-time fallback to batched TOP_K here. |
| 0036 v2 exact ARGSORT rows | Forces chunk_nrows=1. CUB checks nrows==1 before capture status, selecting DeviceRadixSort in either case. |

In evidence/c-tree/ggml/src/ggml-cuda/ggml-cuda.cu,
ggml_cuda_graph_evaluate_and_capture executes the same node dispatch when
running directly or recording. Replay skips host dispatch and uses
cudaGraphLaunch; that alone does not substitute batched kernels for recorded
singleton launches. ggml_backend_cuda_graph_compute waits for stable properties
before capture. With only three sampled tokens, callback evidence cannot test
steady-state 128-token replay behavior.

The scheduler callback branch (llama/ggml/src/ggml-backend.cpp, around 1899)
computes graph views ending at requested tensors and synchronizes each view.
It fragments fusion and changes graph boundaries; it does not explicitly set
the CUDA graph-disable environment variable. Small views may still capture.
Therefore digest equality does not isolate capture from fusion differences.

Actual capture-sensitive fallback found: evidence/c-tree-extra/ggml/src/
ggml-cuda/mmq.cu around 363 avoids the host expert-count readback and stream
sync during capture, walking every bank slot instead of compacting. This is
not a bypass of 0030 MMVQ or 0035's splitting. Wide multi-row CUB ARGSORT uses
DeviceSegmentedSort directly and DeviceSegmentedRadixSort during capture;
0036's forced singleton chunks avoid that particular difference.

Graph compatibility disables capture for MUL_MAT_ID that needs a stream sync,
zero-copy host experts, synchronous host-bank hooks, and op profiling.
The async host-bank option in the window permits capture where compatible.
The observer can also prevent joint fusion by demanding intermediate values.
0035 v2 explicitly disabled fusion and pinned memory but did not explicitly
disable graphs; its earlier successful result cannot identify the cause.
N=1 skips the multi-column branches and retains singleton arithmetic, so its
gold result is compatible with several possible N=2 faults, not proof of one.

The immediate experiment uses GGML_CUDA_DISABLE_GRAPHS=1 for the entire server
process in G3/G4/G5. The controlling code is ggml_cuda_graph::is_enabled() in
evidence/c-tree/ggml/src/ggml-cuda/common.cuh around 1292: any env presence,
including value 0, disables graphs. ggml-cuda.cu consults this predicate in
compute and graph optimization. G1/G2 explicitly start with the variable unset.
No automatic source-level workaround is claimed or applied before measuring.
G3/G4 recovering gold would support a graph-sensitive fault; persistent N2
failure would leave fusion/boundary differences as a live explanation.
G1/G4 measure exact N1 graph cost; G2/G3 compare N2 throughput but can differ
in completion text. G5 measures plain N1 with graphs off, without a fresh
plain N1 graphs-on control. Single-trial timings are diagnostic, not a robust
performance estimate; graph-enabled controls may still reject capture.

The runner enables the existing GGML_CUDA_GRAPH_STATS counter in every leg,
prints its last periodic log entry, and preserves the complete server log.
This does not install an observer callback. If no periodic entry is emitted,
actual capture usage is explicitly reported as unverified.

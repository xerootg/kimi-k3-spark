The c-tree uses llama_context::graph_max_nodes(), not llama_model::max_nodes().
llm_graph_result::reset() sizes its object arena with tensor_overhead * max_nodes;
sched_reserve, resolve_fused_ops and execution all share this bound.
0040 adds L*(64*min(n_seq_max,8)+16) under LLAMA_EXACT_ATTN_COLS=1, before training scaling.
The addition applies even for N1: 0036 selects its singleton branch there too.

Count of allocated GGML tensor objects, per layer per selected stream:
0035 v5/v6: zero. Stack-local tensor copies and CUDA pool allocations are not graph nodes.
0036 v5 indexer: 3 slices*(view+cont)=6; get_k_stream=2; fallback score=11;
top_k=argsort+view=2, plus cont=1: total 22. Fused indexer replaces 11 by 1: total 12.
0036 v5 attention: get_k_stream=2; V/mask/Q views=3; Q/mask cont=2;
select_mask=8 (includes zero tensor and fill); build_attn_mha flash=6,
MLA projection=4, optional transpose/casts<=3; concat<=1: total <=29.
Thus the requested FA graph is <=51 objects/stream/layer; 64 rounds up and also
covers non-FA attention's <=14-object body (<=30 attention, <=52 combined).
The 16/layer allowance covers common/discarded batched views and shared helpers.
We add the full branch count without subtracting baseline objects, so this is
an upper bound rather than a tight incremental live-node count.
V4 uses three fewer indexer packs, two fewer cache views, two fewer attention
packs, and one shared MLA projection: <=42/stream +4/layer; the same bound covers it.
This counts allocations, including views/leaves absent from gf->n_nodes.

0035v6 stats count outer host dispatches once; recursive singleton launches are
excluded. CUDA graph replay does not re-enter host dispatch; counters count
capture/recapture decisions, not executed GPU kernels. Singleton is separate
from batched, so N1 produces useful positive evidence without false batching.
MUL_MAT_ID quantized matvec records the existing 0008 low-level column splitter;
other ID backends record batched. Fused F/Q, routed F/Q and top-k are separate.

fd10 v3 norm-10/norm-4 step-1 reports are alignment artifacts per supplied evidence,
not evidence of engine arithmetic differences. The comparator now associates
explicit singleton graph blocks with each sequence's sample ordinal and rejects
incomplete graph/sample pairings. Raw fd10 v3 TSVs were not supplied here, so
that particular comparison cannot be replayed locally; v4 geometry remains valid.

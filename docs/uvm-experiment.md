# Unified memory on the GH200: can the GPU read offloaded experts without explicit registration?

Experiment of 2026-10-19 on gh200-dev (GH200 480GB, driver 580.105.08, kernel 6.8 with 64 KiB pages,
`nvidia_uvm` with `uvm_ats_mode=1`, HMM disabled). Tool: `tools/uvmbw.cu`
(`nvcc -O3 -arch=sm_90 -o uvmbw uvmbw.cu`, then `./uvmbw MODE REGION_GIB [FILE]`).

The access pattern is one K3 MoE layer: 16 chunks of 6.6 MB at random 64 KiB-aligned offsets in a 16-64 GiB
region, read by a coalesced kernel. "cold" is the first read of new offsets, "warm" the same offsets again.

| mode | what it is | cold GB/s | warm GB/s |
|---|---|---:|---:|
| reg | malloc + `cudaHostRegister` (what the expert store uses) | 345 | 350 |
| ats | plain malloc, no registration, GPU goes through ATS | 343 | 346 |
| thp | malloc + `MADV_HUGEPAGE` (512 MiB PMD pages on this kernel) | 8 | 107 |
| managed | `cudaMallocManaged`, first touch on the CPU, fault-driven migration | 4 | 3143 (now in HBM) |
| managed_hint | managed + PreferredLocation CPU + AccessedBy GPU | 90 | 90 |
| managed_hbm | managed + `cudaMemPrefetchAsync` of the chunks to the GPU | migration 23 GB/s | 3047 |
| file | `mmap` of the expert file, page cache warm, no registration | 346 | 351 |
| file_cold | `mmap`, page cache dropped: the GPU demand-faults from NVMe | 0.4 | 329 |
| file_pf | `mmap`, cold, 32 CPU threads `MADV_POPULATE_READ` the chunks first (8-9 GB/s) | 242 | 346 |
| hbm_ats | plain malloc `mbind` to NUMA node 1 (the HBM), ATS | 2093 | 2776 |
| hbm_reg | same, registered | 2093 | 2856 |
| file_hbm | `mmap`, page cache filled by a thread bound to node 1 | 2163 | 2856 |
| file_pf_hbm | `mmap`, cold, populated by threads bound to node 1 | 737 | 2504 |

`cudaHostRegister` of a read-only file mapping fails (`operation not supported`).

## What this says

- On a GH200, "UVM offload" works at full NVLink-C2C speed as long as the GPU never faults. A plain malloc
  or an mmap of the weight file, with no registration, reads at the same ~345 GB/s as pinned memory.
- The slow paths are the fault-driven ones: GPU demand faults on file pages (0.4 GB/s), fault-driven
  managed-memory migration (4 GB/s), driver prefetch into HBM (23 GB/s), hint-based managed mappings
  (90 GB/s), and 512 MiB transparent huge pages (8 GB/s cold).
- The fix is to keep the faults on the CPU side. Threads that populate the pages the router is about to
  need (`MADV_POPULATE_READ`) reach the same 8-10 GB/s from NVMe as the expert store's O_DIRECT reads.
  After that the GPU reads at link speed.
- The GH200-specific lever is that the HBM is an ordinary NUMA node. Host pages, including page cache,
  placed on node 1 are read by the GPU through the same host pointer at 2.1-2.9 TB/s, which is about 8x
  the C2C rate. A hot-expert tier needs no copies and no second pointer: bind those pages to node 1.

## Consequences for the expert store

- The store's registered LRU bank is equivalent to the "UVM with CPU prefetch" design. There is no speed
  left on the table from not using UVM.
- A part of the bank (or page cache) bound to the HBM node would serve the hottest experts at HBM speed
  and add capacity. With ~30 GB of HBM free (spine 45.5 GB, KV 13.8 GB, compute 3.2 GB), that is
  25 GB on top of the 390 GiB host bank.

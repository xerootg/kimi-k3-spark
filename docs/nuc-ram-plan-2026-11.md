# NUC memory upgrade plan (2026-11-02)

The owner wants all three NUCs at the largest memory that fits, in two waves, ahead of a 10-100 GbE network.
This note records the hardware facts, the parts, the install procedure and what the memory is for.

## What is in the boxes today (read from DMI on 2026-11-02)

| node | model | memory | channels |
|---|---|---|---|
| nuc1 (192.168.4.248) | ASUS NUC 15 Pro NUC15CRKU5, Core Ultra 5 225H, BIOS CRARL579.0025 (2025-06-25) | 1 x 32 GB Crucial CT32G56C46S5 (DDR5-5600, 2R) | **one** |
| nuc2 (192.168.4.249) | same | 2 x 32 GB CT32G56C46S5 | two |
| nuc3 (192.168.4.250) | same | 1 x 32 GB CT32G56C46S5 | **one** |

nuc1 and nuc3 run on a single memory channel: half the DRAM bandwidth, and the iGPU loses its Arc 130T mode (it
reports as "Intel Graphics", 8086:7dd1). The ledger measured nuc3's CPU expert stream at 12 GB/s against nuc2's 27 GB/s.

## Limits of the box

- ASUS's official maximum is 96 GB: 2 x 48 GB DDR5-5600 SO-DIMM, or 2 x 48 GB DDR5-6400 CSO-DIMM.
- 2 x 64 GB (128 GB) works in practice: William Lam ran Crucial 64 GB DDR5-5600 SO-DIMMs in an ASUS NUC 15 Pro
  (Core Ultra 7 255H, same Arrow Lake-H memory controller) with all 128 GB detected. Not on ASUS's list.
- Two slots, so 128 GB is the physical ceiling with today's 64 GB modules.
- DDR5-6400 needs clocked CSO-DIMMs; plain SO-DIMMs run at 5600.

## Parts

Wave 1 (nuc2): **Crucial CT2K64G56C46S5**, 128 GB kit (2 x 64 GB DDR5-5600 SO-DIMM, CL46). This is the kit with
a published NUC 15 Pro result. nuc2's two CT32G56C46S5 sticks then go one each into nuc1 and nuc3, which already
hold the identical part, so both become matched dual-channel 64 GB pairs.

Wave 2 (nuc1, nuc3): the same 128 GB kit for each, if wave 1 passes its tests. That frees four 32 GB sticks.

Why 5600 SO-DIMMs and not 6400 CSO-DIMMs: the measured expert kernels stream 23-27 GB/s per NUC, well under
dual-channel DDR5-5600's ~90 GB/s, so the extra 14 % of 6400 buys nothing until the kernels improve, and no 64 GB
CSO-DIMM has a published NUC 15 Pro result. Capacity is the constraint; speed is not.

Fallback if a NUC does not train 128 GB: 2 x 48 GB (Crucial CT2K48G56C46S5, 96 GB, officially supported). Buy
from a seller with a clean return policy.

## Install procedure (one node at a time)

1. Check ASUS support for a BIOS newer than CRARL579.0025 and flash it first.
2. Cordon and drain the node. nuc3 hosts a Garage node and Jellyfin, and the NUCs carry Longhorn replicas:
   wait for Longhorn volumes to be healthy elsewhere before taking the next node down. Never two NUCs at once.
3. Power off, swap modules, power on. The first boot retrains memory and can take a few minutes on a black screen.
4. Check DMI (`sudo dmidecode -t memory`): both slots populated, 5600 MT/s, full capacity. Check `free -g`.
5. Run a memory test before returning the node to service (MemTest86 from USB, or `memtester` on most of the RAM
   for one pass if the node must stay up).
6. Uncordon; check Longhorn and Garage health; next node.

## What the memory is for, in order of payoff

1. **Dual channel on nuc1 and nuc3 (wave 1, free).** Doubles their DRAM bandwidth and restores the Arc iGPU mode,
   for every workload on them.
2. **A prompt-cache tier for long conversations.** GLM-5.3's KV is ~93 KiB per token, so a 100k-token conversation
   is ~9 GB. The Spark re-reads a 100k prompt in ~15 minutes when its slot is reused; restoring the saved KV from
   NUC RAM takes ~7 s over 10 GbE and under 1 s over 100 GbE (the NUCs' own link is capped by Thunderbolt 4 at
   ~25-32 Gbit/s). Each 128 GB NUC can hold about ten such conversations. This makes a returning turn cost its
   new tokens, not the whole history: the owner's goal of per-turn cost that does not grow with context.
3. **A small fast model for side requests.** gpt-oss-120b MXFP4 (~65 GB) on one NUC would serve Claude Code's
   small-model calls and OpenClaw's utility calls, so they never evict the big model's slot. ~3 GB of weights
   per token at the measured 23-27 GB/s is ~8-9 tok/s on one NUC; enough for titles and summaries.
4. **GLM-5.3 Q2's whole expert set in cluster RAM.** 238 GB of experts: the Spark keeps its hot 78-84 GB; three
   128 GB NUCs (~90-100 GB free each after their services) hold the rest. Each NUC computes the experts it holds
   (shape B in the 2026-10-26 study), so only activations cross the network. Today's verdict on this shape was
   "not worth it" because the Spark has only 2.5 GbE and a ~2.5 ms window per layer. A 10 GbE+ link to the Spark,
   or its ConnectX-7 (present on the Founders Edition, hidden by DGX OS), is what changes that. Expect a modest
   gain (the ledger's estimate was +10-14 % with an ideal link) until the NUC kernels improve.
5. **Faster NUC expert kernels.** The CPU path streams 27 GB/s and the Arc SYCL path 23 GB/s, a third of the
   DRAM's ~90 GB/s. Closing that gap is the lever that makes item 4 pay; only then would DDR5-6400 matter.

Not a fit: GLM-5.3 Q4 (436 GB of experts) does not fit in Spark + 3 x 128 GB with room for services.

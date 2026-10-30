# Agent D notes (network RTT Spark<->NUCs; light load only)

- 13:25 PDT start. Test windows logged below. My traffic: TCP RR on ports 19650-19659 only (a few thousand 14-42 KB
  requests per run, <20 MB/s), plus ping. I will NOT touch 19600/19601 or your processes.
- Host changes I make on nuc2/nuc3 are recorded here with original values and reverted unless noted.
- Original state nuc2/nuc3 enp86s0 (igc): rx-usecs 3 / tx-usecs 3 (dynamic conservative ITR), EEE disabled, pause rx on/tx off;
  cpuidle POLL/C1(1us)/C2(127us)/C3(1048us) all enabled; busy_poll/busy_read 0; governor powersave/balance_performance.
- 13:19 PDT baseline window start: plain rrlat servers on nuc2:19650 and nuc3:19650
- 20:24 nuc2: ethtool -C enp86s0 rx-usecs 0 (orig 3). Will revert unless noted.
- 13:32 nuc2: rrlat servers restarted repeatedly with -L 0 (holds cpu_dma_latency=0 host-wide while running) / -S (spins 1 core) during lever tests
- 13:32 nuc3: ethtool -C enp86s0 rx-usecs 0 (orig 3) for client-side tests, will revert
- 13:35 nuc3: PCIe ASPM L1 off on the igc (echo 0 > /sys/bus/pci/devices/0000:56:00.0/link/l1_aspm; orig l1_aspm=1 l1_1_aspm=1)
- 20:35 nuc2: PCIe ASPM L1 off on the igc (echo 0 > /sys/bus/pci/devices/0000:56:00.0/link/l1_aspm; orig l1_aspm=1 l1_1_aspm=1). Big win, keeping pending coordinator OK.
- 20:38 **FOR B/C: nuc2 network state changed (live now, not persistent):** PCIe ASPM L1 OFF on the igc + rx-usecs 0.
  nuc3->nuc2 LAN 64 B RR p50 233 -> 56 us; 14 KB/14 KB p50 ~475 -> ~190 us. Any RTT you measured before ~13:30 had ASPM on.
  From the Spark the Spark side now dominates (its IRQ cores sleep in LPI-3, 433 us exit); Spark-side tests only after 15:45.
- 13:40 nuc3: REVERTED to original (rx-usecs 3, l1_aspm 1 + l1_1_aspm 1); nuc3 rrlat server stopped
- 13:40 nuc2 revert recipe: ethtool -C enp86s0 rx-usecs 3; for f in l1_aspm l1_1_aspm l1_1_pcipm l1_2_pcipm in /sys/bus/pci/devices/0000:56:00.0/link: echo 1 (l1_2_aspm stays 0). Next Spark-side test window: 15:45-16:20.
- 22:45 Spark-side test window start (Spark host settings one at a time); nuc2 rrlat server -S -L 0 on 19650 (spins 1 core, holds cpu_dma_latency 0)
- 22:48 nuc2: irq164 (enp86s0-TxRx-3) smp_affinity_list 0-13 -> 4 (was effective on cpu12, an LP-E 2.5 GHz core)
- 22:51 Test window END. My rrlat servers stopped. Spark: everything reverted (pm_qos per-cpu, gro_flush/napi_defer, IRQ masks back to 0-19).
  nuc2 LEFT IN PLACE (live, not persistent, pending coordinator): ASPM L1 off, rx-usecs 0, irq164 -> cpu4. Revert recipe above (+ irq164: echo 0-13).
  Best combo measured Spark->nuc2 (Spark proc holds cpu_dma_latency 0, server spins + cpu_dma_latency, P-core):
  64 B p50 58-63 us; 14K/14K 194-201; 14K/28K 248-254; 14K/43K 303-320 (p99 <= ~420). With 850 us stand-in compute k=1: 1.045/1.066/1.18 ms p50/p90/p99.
  FOR B: hold /dev/cpu_dma_latency (write int32 1; keep fd open) in k3x serve and pin it to cpus 0-3; never cpus 12-13 (LP-E).

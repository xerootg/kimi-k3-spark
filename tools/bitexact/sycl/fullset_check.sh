#!/bin/bash
# B's phase-2 runner on nuc2 (run from ~/k3-bitexact/sycl). Usage: b_window.sh refs | full N | split | one IDX PORT [ENV...]
cd ~/k3-bitexact/sycl
BIN=${BIN:-./k3x-v11}
memwait() { until [ $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) -ge ${1:-55} ]; do sleep 3; done; }
run() {   # run IDX PORT LOG [ENV...] : start server, wait for serving/FAIL, print summary lines, leave it running
  local idx=$1 port=$2 log=$3; shift 3
  (setsid nohup sudo -n -u xero -g render env "$@" LD_LIBRARY_PATH=lib taskset -c 4-11 choom -n 1000 -- $BIN $idx ../remote/experts.bin - - serveidx $port > $log 2>&1 &)
  until grep -q "serving on\|FAIL\|rror" $log; do sleep 2; if ! pgrep -f "k3x-v1[01] $idx" >/dev/null; then break; fi; done
  grep -E "loaded|ROWCHECK|FULLCHECK|self|FAIL|reached|rror" $log | grep -v "^ioctl" | sed "s|^|  [$port] |"
}
stopport() { for p in $(pgrep -a k3x-v1[01] | awk -v P="serveidx $1" 'index($0,P){print $1}'); do sudo -n kill $p; done; sleep 3; }
TE=/sys/kernel/tracing/events/i915
traceon()  { sudo -n sh -c "echo > /sys/kernel/tracing/trace; for e in i915_gem_shrink i915_gem_evict i915_gem_evict_node i915_gem_evict_vm i915_vma_unbind; do echo 1 > $TE/\$e/enable; done; echo 1 > /sys/kernel/tracing/tracing_on"; }
traceoff() { sudo -n sh -c "for e in i915_gem_shrink i915_gem_evict i915_gem_evict_node i915_gem_evict_vm i915_vma_unbind; do echo 0 > $TE/\$e/enable; done"; sudo -n cat /sys/kernel/tracing/trace | grep -o -E "i915_[a-z_]+:" | sort | uniq -c; }
case $1 in
diag)   # one full cold start with i915 tracepoints and NEO exec printing
  memwait 55; traceon; echo "diag $(date +%T)"
  run ../remote/experts.idx 19613 /tmp/b_diag.log K3X_MAX_GB=49 K3X_FULLCHECK=1 NEOReadDebugKeys=1 PrintBOsForSubmit=1 $EXTRAENV
  traceoff; stopport 19613 ;;
refs)
  rm -f rows-ref.bin; total=$(wc -l < ../remote/experts.idx)
  sed -n 1,2400p ../remote/experts.idx > /tmp/b_p1.idx; sed -n 2401,4800p ../remote/experts.idx > /tmp/b_p2.idx; sed -n 4801,${total}p ../remote/experts.idx > /tmp/b_p3.idx
  for p in 1 2 3; do memwait 50; echo "ref part $p $(date +%T)"; run /tmp/b_p$p.idx 19613 /tmp/b_ref$p.log K3X_MAX_GB=40 K3X_ROWSOUT=$PWD/rows-ref.bin; stopport 19613; done
  ls -la rows-ref.bin ;;
full)
  for i in $(seq 1 ${2:-5}); do memwait 55; echo "full cold start $i $(date +%T) MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GB"
    run ../remote/experts.idx 19613 /tmp/b_full$i.log K3X_MAX_GB=49 K3X_FULLCHECK=1 $EXTRAENV; stopport 19613; done ;;
split)
  memwait 55; total=$(wc -l < ../remote/experts.idx)
  sed -n 1,3500p ../remote/experts.idx > /tmp/b_s1.idx; sed -n 3501,${total}p ../remote/experts.idx > /tmp/b_s2.idx
  echo "split $(date +%T)"; run /tmp/b_s1.idx 19613 /tmp/b_s1.log K3X_MAX_GB=40 K3X_FULLCHECK=1
  run /tmp/b_s2.idx 19614 /tmp/b_s2.log K3X_MAX_GB=40 K3X_FULLCHECK=1
  echo "both up; MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GB" ;;
stop) stopport 19613; stopport 19614 ;;
esac

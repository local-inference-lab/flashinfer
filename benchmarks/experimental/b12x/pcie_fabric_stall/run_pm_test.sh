#!/bin/bash
# Approved 2026-10-29 (option 1): HSMP power efficiency mode 2 (IO performance) and DF C-state off,
# each for one measurement, restored to the read-back originals (power 0, dfc 1). A/P/D/A.
set -u
cd /data/ssd2/claude-ds41/work/p2pstall
export CUDA_DEVICE_ORDER=PCI_BUS_ID
G=${GPUS:-1,5,9,13}
OUT=dumps-pm2
mkdir -p $OUT
S=$OUT/state.log
measure() {  # $1 = tag
  ./p2pstall --gpus $G --iters 700000 --ctas 16 --kib 80 --patterns read_ring,pull --dump $OUT/$1 > $OUT/$1-read.log 2>&1
  ./p2pstall --gpus $G --iters 30000 --ctas 16 --kib 80 --patterns write_all,push > $OUT/$1-write.log 2>&1
}
restore() { ./hsmpset_pm power 0 >> $S 2>&1; ./hsmpset_pm dfc 1 >> $S 2>&1; ./hsmpget >> $S 2>&1; ./hsmpq >> $S 2>&1; }
{ echo "start $(date -u +%T.%N)"; ./hsmpget; ./hsmpq; } > $S 2>&1
measure base1
trap restore EXIT
echo "power 2 $(date -u +%T.%N)" >> $S
./hsmpset_pm power 2 >> $S 2>&1 && { ./hsmpq >> $S 2>&1; measure power2; }
echo "power 2 end $(date -u +%T.%N)" >> $S
./hsmpset_pm power 0 >> $S 2>&1
echo "dfc 0 $(date -u +%T.%N)" >> $S
./hsmpset_pm dfc 0 >> $S 2>&1 && measure dfc0
echo "dfc 0 end $(date -u +%T.%N)" >> $S
restore
trap - EXIT
measure base2
echo "done $(date -u +%T.%N)" >> $S

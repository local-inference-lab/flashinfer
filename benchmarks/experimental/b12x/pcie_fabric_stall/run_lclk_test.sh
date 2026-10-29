#!/bin/bash
# Approved 2026-10-29: pin LCLK DPM to level 1 on NBIO 0-3 (HSMP), measure, restore 0..1.
# A/B/A: baseline (dynamic 0..1) -> pinned (1..1) -> baseline again. Pinned only during phase B.
set -u
cd /data/ssd2/claude-ds41/work/p2pstall
export CUDA_DEVICE_ORDER=PCI_BUS_ID
G=${GPUS:-1,5,9,13}
OUT=dumps-lclk
mkdir -p $OUT
measure() {  # $1 = tag
  ./p2pstall --gpus $G --iters 700000 --ctas 16 --kib 80 --patterns read_ring,pull --dump $OUT/$1 > $OUT/$1-read.log 2>&1
  ./p2pstall --gpus $G --iters 30000 --ctas 16 --kib 80 --patterns write_all,push > $OUT/$1-write.log 2>&1
}
{ echo "A start $(date -u +%T.%N)"; ./hsmpq; } > $OUT/state.log 2>&1
measure base1
restore() { ./hsmpset_lclk 0 1 >> $OUT/state.log 2>&1; echo "restored $(date -u +%T.%N)" >> $OUT/state.log; ./hsmpq >> $OUT/state.log 2>&1; }
trap restore EXIT
echo "pin $(date -u +%T.%N)" >> $OUT/state.log
./hsmpset_lclk 1 1 >> $OUT/state.log 2>&1
./hsmpq >> $OUT/state.log 2>&1
measure pinned
echo "pinned end $(date -u +%T.%N)" >> $OUT/state.log
restore
trap - EXIT
measure base2
echo "done $(date -u +%T.%N)" >> $OUT/state.log

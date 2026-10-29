#!/bin/bash
# Cost of phased push on a single-quadrant set (13-16) versus cross-quadrant (1,5,9,13).
set -u
cd /data/ssd2/claude-ds41/work/p2pstall
export CUDA_DEVICE_ORDER=PCI_BUS_ID
OUT=dumps-phased
mkdir -p $OUT
for G in 13,14,15,16 1,5,9,13; do
  ./p2pstall --gpus $G --iters 300000 --ctas 8,16 --kib 16,80 --patterns pull,push_phased,rsag_phased,ring > $OUT/g${G//,/-}-a.log 2>&1
  ./p2pstall --gpus $G --iters 30000 --ctas 8,16 --kib 16,80 --patterns push,rsag_push > $OUT/g${G//,/-}-b.log 2>&1
done
echo done > $OUT/done

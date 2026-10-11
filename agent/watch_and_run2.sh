#!/bin/bash
OUT=/root/autodl-tmp/bench_out
while true; do
  if grep -q MEM1_DONE $OUT/mem_stage1.out 2>/dev/null; then break; fi
  sleep 60
done
sleep 30
cd /root/autodl-tmp/nidus_agent
/root/miniconda3/bin/python3 bench_runner_mem2.py > $OUT/mem_stage2.out 2>&1

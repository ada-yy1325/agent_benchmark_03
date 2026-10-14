#!/bin/bash
# 2.4 final verification: c=32,48,64,128 on batched=8192, max-seqs=64
# Run via: nohup bash run_final_bench.sh > _all.log 2>&1 &
OUTD=/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/perf_opt/results/sched/final
BENCH=/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/perf_opt/scripts/bench_serving.py
MODELDIR=/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-self
PY=/opt/mamba/bin/python3

for C in 32 48 64 128; do
  echo "=== c=$C ==="
  echo "start: $(date)"
  $PY $BENCH \
    --served-name dsv4 \
    --model-dir "$MODELDIR" \
    --base-url http://127.0.0.1:8000/v1 \
    --input-len 4096 --output-len 4096 \
    --concurrency $C \
    --num-requests 128 \
    --out "$OUTD/final_c${C}_v2.json" \
    --case-id "final_c${C}_v2" \
    --npu-stats "$OUTD/final_c${C}_v2_npu.csv" \
    --temp 0.0 --seed 1
  echo "end: $(date)"
  echo ""
done
echo "ALL DONE: $(date)"
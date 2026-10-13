#!/bin/bash
# baseline_c1_x3.sh - Run 3 baseline c=1 runs sequentially
set -e
cd /inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/perf_opt
mkdir -p results/baseline npu_stats

for run in 1 2 3; do
    echo "===== Run $run / 3 ====="
    python3 scripts/bench_serving.py \
        --base-url http://127.0.0.1:8000/v1 \
        --served-name dsv4 \
        --model-dir /inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-self \
        --input-len 4096 \
        --output-len 1024 \
        --concurrency 1 \
        --num-requests 32 \
        --seed $((42 + run)) \
        --case-id baseline_c1_run${run} \
        --out results/baseline/baseline_c1_run${run}.json \
        --npu-stats npu_stats/baseline_c1_run${run}.csv \
        2>&1 | tee results/baseline/baseline_c1_run${run}.log
    echo "===== Run $run done ====="
done
echo "===== ALL 3 RUNS COMPLETE ====="
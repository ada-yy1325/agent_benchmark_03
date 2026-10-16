#!/bin/bash
# Small-scale test for corrected Base GSM8K evaluation
set -e

cd /inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test || exit 1

echo "=== Step 1: Kill old processes ==="
pkill -f 'vllm.*8802' 2>/dev/null || true
sleep 1

echo "=== Step 2: Pull latest code ==="
git fetch origin
git reset --hard origin/main

echo "=== Step 3: Start vLLM FP16 server ==="
nohup python3 start_vllm_base_fp16.py > vllm_fp16_base.log 2>&1 &
VPID=$!
echo "vLLM PID: $VPID"

echo "=== Step 4: Wait for server ready ==="
for i in $(seq 1 120); do
  if curl -s http://localhost:8802/health > /dev/null 2>&1; then
    echo "Server ready after ${i}x5s"
    break
  fi
  sleep 5
done

echo "=== Step 5: Run small-scale test (20 questions) ==="
python3 eval_gsm8k_base_correct.py --max-questions 20

echo "=== DONE ==="

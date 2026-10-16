#!/bin/bash
# W8A8 GSM8K Eval — 一键启动脚本（nohup 后台运行）
# 用法: nohup bash start_eval_w8a8.sh > w8a8_setup.log 2>&1 &
set -e

cd /inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test

# Kill old servers
echo "[$(date '+%H:%M:%S')] Killing old vLLM servers..."
pkill -f 'vllm.*880[23]' 2>/dev/null || true
sleep 2

# Clean up old results
rm -f eval_w8a8.log eval_gsm8k_base_correct_results_w8a8.json

# Start W8A8 vLLM server in tmux
echo "[$(date '+%H:%M:%S')] Starting W8A8 vLLM server (port 8803)..."
tmux kill-session -t vllm_w8a8 2>/dev/null || true
tmux new-session -d -s vllm_w8a8
tmux send-keys -t vllm_w8a8 "cd /inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test && python3 start_vllm_base_w8a8.py" Enter

# Wait for server ready (up to 5 min)
echo "[$(date '+%H:%M:%S')] Waiting for W8A8 server..."
for i in $(seq 1 60); do
  sleep 5
  if curl -s http://127.0.0.1:8803/v1/models > /dev/null 2>&1; then
    echo "[$(date '+%H:%M:%S')] W8A8 server ready after ${i}x5s"
    break
  fi
  if [ $i -eq 60 ]; then
    echo "[$(date '+%H:%M:%S')] ERROR: W8A8 server failed to start"
    tmux capture-pane -t vllm_w8a8 -p | tail -20
    exit 1
  fi
done

# Run eval in background
echo "[$(date '+%H:%M:%S')] Starting GSM8K eval (W8A8)..."
python3 -u eval_gsm8k_base_correct.py \
  --port 8803 \
  > eval_w8a8.log 2>&1 &
EVAL_PID=$!
echo "[$(date '+%H:%M:%S')] Eval PID: $EVAL_PID"

# Wait for completion
echo "[$(date '+%H:%M:%S')] Waiting for eval to finish (this will take ~75 min)..."
wait $EVAL_PID 2>/dev/null || true

echo "[$(date '+%H:%M:%S')] === EVAL DONE ==="
tail -15 eval_w8a8.log
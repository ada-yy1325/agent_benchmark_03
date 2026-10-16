#!/bin/bash
# ──────────────────────────────────────────────────────────
# DeepSeek-V4-Flash-w8a8-mtp 一键评测流程
# 用法: bash run_dsv4_eval.sh [smoke|full]
# ──────────────────────────────────────────────────────────
set -e

MODE="${1:-smoke}"
BASE_DIR="/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test"

echo "=========================================="
echo " DeepSeek-V4-Flash-w8a8-mtp 评测"
echo " Mode: $MODE"
echo "=========================================="

# Clean up any existing vLLM on port 8000
echo ""
echo "[Step 0] Clean up..."
pkill -f "vllm.*8000" 2>/dev/null || true
sleep 2

echo ""
echo "[Step 1] Starting vLLM server in tmux session 'dsv4'..."
tmux new-session -d -s dsv4 -c "$BASE_DIR" "python3 start_dsv4_w8a8.py 2>&1 | tee /tmp/dsv4_server.log"
echo "  tmux session 'dsv4' created"
echo "  Log: /tmp/dsv4_server.log"

echo ""
echo "[Step 2] Waiting for server to be ready..."
for i in $(seq 1 120); do
    if curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
        echo "  ✅ Server ready after ${i}s"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "  ❌ Server failed to start within 120s. Check log:"
        tmux capture-pane -t dsv4 -p | tail -30
        exit 1
    fi
    sleep 5
done

echo ""
echo "[Step 3] Running smoke test (5 queries)..."
cd "$BASE_DIR"
python3 quick_test_dsv4.py

if [ "$MODE" = "full" ]; then
    echo ""
    echo "[Step 4] Running FULL GPQA-Diamond evaluation (198 questions)..."
    python3 eval_gpqa_dsv4.py --temperature 0
    echo ""
    echo "✅ Full eval complete! Results in dsv4_gpqa_result.json"
else
    echo ""
    echo "[Step 4] Running smoke GPQA evaluation (5 questions)..."
    python3 eval_gpqa_dsv4.py --smoke 5 --temperature 0
    echo ""
    echo "✅ Smoke test complete!"
    echo "Run 'bash run_dsv4_eval.sh full' for full 198-question eval."
fi
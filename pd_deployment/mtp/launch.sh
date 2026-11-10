#!/bin/bash
# ===========================================================================
# P/D Separation Launch - MTP ON Version
# Prefill (NPU 0-7) + Decode (NPU 8-15) via Mooncake KV Transfer
# MTP: num_speculative_tokens=1, enforce_eager=true
# ===========================================================================
set -e

REPO="/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test"
TMP="/tmp/pd_test"

echo "╔══════════════════════════════════════════╗"
echo "║  P/D Separation — MTP ON Launch         ║"
echo "╚══════════════════════════════════════════╝"

# 1. Pull latest code
echo "[1/4] Pulling latest code..."
cd "$REPO" && git fetch origin && git reset --hard origin/main

# 2. Kill old processes
echo "[2/4] Killing old vllm processes..."
pkill -f "vllm serve" 2>/dev/null || true
sleep 3

# 3. Regenerate MTP args + launch scripts
echo "[3/4] Regenerating MTP launch scripts..."
python3 perf_opt/scripts/regenerate_pd_mtp.py

# 4. Start Decode first via tmux, then Prefill
echo "[4/4] Starting Decode → Prefill (MTP ON)..."
tmux kill-server 2>/dev/null; sleep 1

# Launch decode in tmux
tmux new-session -d -s pd_decode bash
sleep 0.5
tmux send-keys -t pd_decode "bash ${TMP}/_pd_decode_mtp_launch.sh" Enter
echo "   Decode tmux 'pd_decode' started (devices=8,9,10,11,12,13,14,15)"

# Wait for decode healthy
echo "   Waiting for decode :8001..."
for i in $(seq 1 180); do
  if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8001/health 2>/dev/null | grep -q 200; then
    echo "   ✅ Decode healthy after ~${i}0s"
    break
  fi
  sleep 10
done

# Launch prefill in tmux
tmux new-session -d -s pd_prefill bash
sleep 0.5
tmux send-keys -t pd_prefill "bash ${TMP}/_pd_prefill_mtp_launch.sh" Enter
echo "   Prefill tmux 'pd_prefill' started (devices=0,1,2,3,4,5,6,7)"

# Wait for prefill healthy
echo "   Waiting for prefill :8000..."
for i in $(seq 1 180); do
  if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health 2>/dev/null | grep -q 200; then
    echo "   ✅ Prefill healthy after ${i}0s"
    break
  fi
  sleep 10
done

echo ""
echo "✅ P/D MTP ON launch complete!"
echo "   Prefill: http://127.0.0.1:8000 (NPU 0-7, kv_producer, MTP)"
echo "   Decode:  http://127.0.0.1:8001 (NPU 8-15, kv_consumer, MTP)"
echo ""
echo "Verify: python3 pd_deployment/scripts/verify_pd.py --mtp"
echo "Bench:  python3 pd_deployment/scripts/run_benchmark.py --mtp"
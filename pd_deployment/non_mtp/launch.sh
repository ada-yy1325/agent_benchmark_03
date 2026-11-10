#!/bin/bash
# ===========================================================================
# P/D Separation Launch - No-MTP Version
# Prefill (NPU 0-7) + Decode (NPU 8-15) via Mooncake KV Transfer
# ===========================================================================
set -e

REPO="/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test"
TMP="/tmp/pd_test"

echo "╔══════════════════════════════════════════╗"
echo "║  P/D Separation — No-MTP Launch         ║"
echo "╚══════════════════════════════════════════╝"

# 1. Pull latest code
echo "[1/4] Pulling latest code..."
cd "$REPO" && git fetch origin && git reset --hard origin/main

# 2. Kill old processes
echo "[2/4] Killing old vllm processes..."
pkill -f "vllm serve" 2>/dev/null || true
sleep 3

# 3. Regenerate No-MTP args + launch scripts
echo "[3/4] Regenerating No-MTP launch scripts..."
python3 perf_opt/scripts/regenerate_pd.py

# 4. Start Decode first (wait for it), then Prefill
echo "[4/4] Starting Decode → Prefill..."
bash "${TMP}/_pd_decode_launch.sh" &
DECODE_PID=$!
echo "   Decode PID=$DECODE_PID"

# Wait for decode healthy
echo "   Waiting for decode :8001..."
for i in $(seq 1 180); do
  if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8001/health 2>/dev/null | grep -q 200; then
    echo "   ✅ Decode healthy after ${i}0s"
    break
  fi
  sleep 10
done

# Start Prefill
bash "${TMP}/_pd_prefill_launch.sh" &
PREFILL_PID=$!
echo "   Prefill PID=$PREFILL_PID"

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
echo "✅ P/D No-MTP launch complete!"
echo "   Prefill: http://127.0.0.1:8000 (NPU 0-7, kv_producer)"
echo "   Decode:  http://127.0.0.1:8001 (NPU 8-15, kv_consumer)"
echo ""
echo "Verify: python3 pd_deployment/scripts/verify_pd.py"
echo "Bench:  python3 pd_deployment/scripts/run_benchmark.py --no-mtp"
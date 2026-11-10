#!/bin/bash
# ===========================================================================
# P/D Separation Launch - No-MTP Version
# Prefill (NPU 0-7) + Decode (NPU 8-15) via Mooncake KV Transfer
# Uses start_pd_mooncake.py --serial (tmux-based, survives exec)
# ===========================================================================
set -e

REPO="/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test"

echo "╔══════════════════════════════════════════╗"
echo "║  P/D Separation — No-MTP Launch         ║"
echo "╚══════════════════════════════════════════╝"

cd "$REPO" && git fetch origin && git reset --hard origin/main
echo "[1/2] Code pulled to latest"

echo "[2/2] Starting PD via orchestrator (tmux, D→P serial)..."
python3 perf_opt/scripts/start_pd_mooncake.py --serial

echo ""
echo "✅ P/D No-MTP launch complete!"
echo "   Prefill: http://127.0.0.1:8000 (NPU 0-7, kv_producer)"
echo "   Decode:  http://127.0.0.1:8001 (NPU 8-15, kv_consumer)"
echo ""
echo "Verify: python3 pd_deployment/scripts/verify_pd.py"
echo "Bench:  python3 pd_deployment/scripts/run_benchmark.py --no-mtp"
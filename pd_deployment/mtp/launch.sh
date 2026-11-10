#!/bin/bash
# ===========================================================================
# P/D Separation Launch - MTP ON Version
# Prefill (NPU 0-7) + Decode (NPU 8-15) via Mooncake KV Transfer
# MTP: num_speculative_tokens=1, enforce_eager=true
# Uses start_pd_mooncake.py --serial --mtp
# ===========================================================================
set -e

REPO="/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test"

echo "╔══════════════════════════════════════════╗"
echo "║  P/D Separation — MTP ON Launch         ║"
echo "╚══════════════════════════════════════════╝"

cd "$REPO" && git fetch origin && git reset --hard origin/main
echo "[1/2] Code pulled to latest"

echo "[2/2] Starting PD+MTP via orchestrator (tmux, D→P serial)..."
python3 perf_opt/scripts/start_pd_mooncake.py --serial --mtp

echo ""
echo "✅ P/D MTP ON launch complete!"
echo "   Prefill: http://127.0.0.1:8000 (NPU 0-7, kv_producer, MTP)"
echo "   Decode:  http://127.0.0.1:8001 (NPU 8-15, kv_consumer, MTP)"
echo ""
echo "Verify: python3 pd_deployment/scripts/verify_pd.py --mtp"
echo "Bench:  python3 pd_deployment/scripts/run_benchmark.py --mtp"
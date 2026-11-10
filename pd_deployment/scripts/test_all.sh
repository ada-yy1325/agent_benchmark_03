#!/bin/bash
# ===========================================================================
# P/D Deployment — Full End-to-End Verification Pipeline
# Tests both No-MTP and MTP versions: launch → verify → smoke bench
# ===========================================================================
set -e

REPO="/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test"
PD_DIR="$REPO/pd_deployment"

echo "╔══════════════════════════════════════════════════╗"
echo "║  P/D Deployment E2E Verification Pipeline       ║"
echo "║  Tests: No-MTP & MTP → verify → smoke bench    ║"
echo "╚══════════════════════════════════════════════════╝"

cd "$REPO"
git fetch origin && git reset --hard origin/main

PASS=0
FAIL=0

# ─── Function: test one version ─────────────────────────────────────────
test_version() {
  local TAG="$1"   # "non_mtp" or "mtp"
  local MTP_FLAG="$2"  # "" or "--mtp"
  local LABEL="$3" # "No-MTP" or "MTP ON"
  
  echo ""
  echo "──────────────────────────────────────────────────"
  echo "  Testing: P/D $LABEL"
  echo "──────────────────────────────────────────────────"
  
  # 1. Launch
  echo "[1/3] Launching $LABEL..."
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 3
  bash "$PD_DIR/$TAG/launch.sh"
  
  # 2. Verify
  echo "[2/3] Verifying $LABEL..."
  if python3 "$PD_DIR/scripts/verify_pd.py" $MTP_FLAG; then
    echo "  ✅ $LABEL verification PASSED"
  else
    echo "  ❌ $LABEL verification FAILED"
    FAIL=$((FAIL + 1))
    return 1
  fi
  
  # 3. Quick benchmark
  echo "[3/3] Smoke benchmark $LABEL..."
  if python3 "$PD_DIR/scripts/run_benchmark.py" --quick $MTP_FLAG; then
    echo "  ✅ $LABEL benchmark PASSED"
    PASS=$((PASS + 1))
  else
    echo "  ❌ $LABEL benchmark FAILED"
    FAIL=$((FAIL + 1))
  fi
}

# ─── Run both versions ──────────────────────────────────────────────────
test_version "non_mtp" "" "No-MTP"
test_version "mtp" "--mtp" "MTP ON"

# ─── Summary ────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  E2E Pipeline Results                           ║"
echo "║  ✅ $PASS passed  ❌ $FAIL failed                 ║"
echo "╚══════════════════════════════════════════════════╝"

python3 "$REPO/perf_opt/scripts/pd_read_results.py"
echo ""
echo "Results saved: /tmp/pd_bench/pd_*.json"
echo "Snapshots available: perf_opt/summary/pd_mtp_bench_summary.md"

exit $FAIL
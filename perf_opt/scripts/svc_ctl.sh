#!/usr/bin/env bash
# ===========================================================================
# svc_ctl.sh — SLO benchmark service lifecycle manager
# 
# 约束：vLLM 服务启动必须从宿主机 shell 执行（exec 容器 multiprocessing
# worker 子进程无法访问 NPU 设备，报 rtSetDevice error 107001）。
# 压测和分析由 exec 自主完成。
# 
# 用法（在宿主机 shell 中）：
#   bash perf_opt/scripts/svc_ctl.sh up_single   # 单服 TP8+MTP (8卡, :8000)
#   bash perf_opt/scripts/svc_ctl.sh up_dual     # 双服 TP8+MTP (16卡, :8000 + :8001)
#   bash perf_opt/scripts/svc_ctl.sh down        # 全部停止
#   bash perf_opt/scripts/svc_ctl.sh status      # 检查端口/tmux/进程
# ===========================================================================
set -euo pipefail

REPO="/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test"
MODEL_PATH="$REPO/models/DeepSeek-V4-Flash-w8a8-npu16"
SCRIPT="$REPO/perf_opt/scripts/_serve_tp8_mtp.sh"

cd "$REPO"

# ── helpers ──
log()  { echo "[svc_ctl] $(date +%H:%M:%S) $*"; }
health() {
  local p=$1
  curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${p}/health" 2>/dev/null || echo "000"
}

# ── down ──
down() {
  log "stopping all services..."
  tmux kill-session -t single_tp8_a 2>/dev/null || true
  tmux kill-session -t single_tp8_b 2>/dev/null || true
  tmux kill-session -t single_tp8   2>/dev/null || true
  tmux kill-session -t pd_decode    2>/dev/null || true
  tmux kill-session -t pd_prefill   2>/dev/null || true
  pkill -f start_server_phase5 2>/dev/null || true
  # 按端口杀残留
  for p in 8000 8001; do
    ss -tlnp 2>/dev/null | grep ":${p} " | awk '{print $NF}' | cut -d, -f2 | cut -d= -f2 | xargs -r kill 2>/dev/null || true
  done
  sleep 2
  log "down complete"
}

# ── status ──
status() {
  echo "=== PORTS ==="
  for p in 8000 8001; do
    echo "  :${p} -> $(health $p)"
  done
  echo "=== TMUX ==="
  tmux ls 2>&1 || echo "  (no sessions)"
  echo "=== VLLM PROCS ==="
  ps aux 2>/dev/null | grep -c "[s]tart_server_phase5" || echo "  0"
}

# ── up_single ──
up_single() {
  down
  log "starting single TP8+MTP (devices 0-7, port 8000)..."
  ASCEND_RT_VISIBLE_DEVICES=0-7 bash "$SCRIPT" \
    --port 8000 --log /tmp/single_tp8.log --mtp 1
  log "single TP8+MTP ready on :8000"
}

# ── up_dual ──
up_dual() {
  down
  log "starting dual TP8+MTP — instance A (devices 0-7, port 8000)..."
  ASCEND_RT_VISIBLE_DEVICES=0-7 bash "$SCRIPT" \
    --port 8000 --log /tmp/dual_a.log --mtp 1

  log "starting dual TP8+MTP — instance B (devices 8-15, port 8001)..."
  ASCEND_RT_VISIBLE_DEVICES=8-15 bash "$SCRIPT" \
    --port 8001 --log /tmp/dual_b.log --mtp 1 &
  
  # wait for B
  for i in $(seq 1 60); do
    if [ "$(health 8001)" = "200" ]; then
      log "dual TP8+MTP both ready (:8000 + :8001)"
      return 0
    fi
    sleep 10
  done
  log "ERROR: dual TP8+MTP instance B startup timeout"
  return 1
}

# ── dispatch ──
case "${1:-}" in
  up_single) up_single ;;
  up_dual)   up_dual ;;
  down)      down ;;
  status)    status ;;
  *)
    echo "Usage: $0 {up_single|up_dual|down|status}"
    exit 1
    ;;
esac
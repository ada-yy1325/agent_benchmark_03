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
cd "$REPO"

# ── helpers ──
log()  { echo "[svc_ctl] $(date +%H:%M:%S) $*"; }
health() {
  local p=$1
  curl -s -o /dev/null --connect-timeout 3 -w "%{http_code}" "http://127.0.0.1:${p}/health" 2>/dev/null || echo "000"
}

# ── gen_launch_script: 生成 tmux 启动脚本（复刻 PD 已验证模式）──
# PD 部署成功关键：单独生成 bash 脚本，显式 export 所有 env vars，
# 在 tmux session 中执行，确保 multiprocessing worker 继承完整 NPU 环境。
gen_launch_script() {
  local name="$1"    # e.g. single_a, dual_b
  local devices="$2" # e.g. 0,1,2,3,4,5,6,7
  local port="$3"
  local mtp="$4"     # 0 or 1
  local logfile="$5"

  local mtp_block=""
  local spec_cfg='{"method":"mtp","num_speculative_tokens":1,"enforce_eager":true}'
  if [ "$mtp" = "1" ]; then
    mtp_block="--speculative-config '${spec_cfg}' \\"
  fi

  local add_cfg='{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"enable_dsa_cp":true,"multistream_overlap_shared_expert":true}'

  cat > "/tmp/_${name}_launch.sh" << LAUNCH_SCRIPT_EOF
#!/bin/bash
# Auto-generated launch script for ${name} (TP8, port ${port})
set -euo pipefail

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export HCCL_INTRA_ROCE_ENABLE=1
export ASCEND_BUFFER_POOL=4:8
export ASCEND_RT_VISIBLE_DEVICES=${devices}
export ASCEND_VISIBLE_DEVICES=${devices}
export COMBINED_ENABLE=1
export ACL_OP_COMPILER_CACHE_MODE=enable
export ACL_OP_COMPILER_CACHE_DIR=/tmp/_aclopcache_${name}

echo "=== \$(date) : TP8 server ${name} starting (devices ${devices}, port ${port}) ==="

MODEL="/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-npu16"

python3 -u -c "
import json, os, sys
env_vars = {k: os.environ.get(k,'') for k in ['ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES','LD_LIBRARY_PATH','PYTHONPATH']}
print(f'[preflight] ASCEND_RT_VISIBLE_DEVICES={env_vars[\"ASCEND_RT_VISIBLE_DEVICES\"]}', flush=True)
print(f'[preflight] ASCEND_VISIBLE_DEVICES={env_vars[\"ASCEND_VISIBLE_DEVICES\"]}', flush=True)
import torch_npu; print(f'[preflight] torch_npu device_count={torch_npu.npu.device_count()}', flush=True)
" 2>&1

echo "=== vLLM serve launch ==="

vllm serve "\${MODEL}" \\
  --host 0.0.0.0 --port ${port} \\
  --max-model-len 133120 \\
  --max-num-batched-tokens 8192 \\
  --served-model-name dsv4 \\
  --gpu-memory-utilization 0.9 \\
  --max-num-seqs 64 \\
  --data-parallel-size 1 \\
  --tensor-parallel-size 8 \\
  --enable-expert-parallel \\
  --quantization ascend \\
  --block-size 128 \\
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \\
  --tokenizer-mode deepseek_v4 \\
  --tool-call-parser deepseek_v4 \\
  --enable-auto-tool-choice \\
  --reasoning-parser deepseek_v4 \\
  --additional-config '${add_cfg}' \\
  ${mtp_block}
  2>&1 | tee ${logfile}
LAUNCH_SCRIPT_EOF
  chmod +x "/tmp/_${name}_launch.sh"
}

# ── down ──
down() {
  log "stopping all services..."
  tmux kill-session -t single_tp8 2>/dev/null || true
  tmux kill-session -t dual_a 2>/dev/null || true
  tmux kill-session -t dual_b 2>/dev/null || true
  tmux kill-session -t pd_decode    2>/dev/null || true
  tmux kill-session -t pd_prefill   2>/dev/null || true
  pkill -f start_server_phase5 2>/dev/null || true
  pkill -f "vllm serve"       2>/dev/null || true
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
  log "starting single TP8+MTP (devices 0-7, port 8000) via tmux..."
  gen_launch_script "single_tp8" "0,1,2,3,4,5,6,7" "8000" "1" "/tmp/single_tp8.log"

  tmux new-session -d -s single_tp8 "bash /tmp/_single_tp8_launch.sh"
  sleep 1
  log "launched tmux session 'single_tp8'"

  log "waiting for :8000/health..."
  for i in $(seq 1 60); do
    if [ "$(health 8000)" = "200" ]; then
      log "single TP8+MTP ready on :8000"
      return 0
    fi
    sleep 10
    echo "  ... $((i*10))s"
  done
  log "ERROR: single TP8+MTP startup timeout (600s). Check /tmp/single_tp8.log"
  return 1
}

# ── up_dual ──
up_dual() {
  down
  log "starting dual TP8+MTP — instance A (devices 0-7, port 8000) via tmux..."
  gen_launch_script "dual_a" "0,1,2,3,4,5,6,7" "8000" "1" "/tmp/dual_a.log"
  tmux new-session -d -s dual_a "bash /tmp/_dual_a_launch.sh"
  sleep 1
  log "launched tmux session 'dual_a'"

  log "waiting for :8000/health..."
  for i in $(seq 1 60); do
    if [ "$(health 8000)" = "200" ]; then break; fi
    sleep 10
    echo "  ... :8000 $((i*10))s"
  done
  if [ "$(health 8000)" != "200" ]; then
    log "ERROR: dual instance A startup timeout"
    return 1
  fi
  log "dual instance A ready on :8000"

  log "starting dual TP8+MTP — instance B (devices 8-15, port 8001) via tmux..."
  gen_launch_script "dual_b" "8,9,10,11,12,13,14,15" "8001" "1" "/tmp/dual_b.log"
  tmux new-session -d -s dual_b "bash /tmp/_dual_b_launch.sh"
  sleep 1
  log "launched tmux session 'dual_b'"

  log "waiting for :8001/health..."
  for i in $(seq 1 60); do
    if [ "$(health 8001)" = "200" ]; then
      log "dual TP8+MTP both ready (:8000 + :8001)"
      return 0
    fi
    sleep 10
    echo "  ... :8001 $((i*10))s"
  done
  log "ERROR: dual instance B startup timeout"
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
#!/usr/bin/env bash
# 远程启动 TP8 MTP decode 实例（PD 分离架构中的 decode 侧）
# 用法（远程）:
#   ASCEND_RT_VISIBLE_DEVICES=0-7 bash _serve_tp8_mtp.sh --port 8001 --log /tmp/pd_test/decode_v2.log
#
# 也可用作独立单 TP8 服务（非 PD 架构）:
#   ASCEND_RT_VISIBLE_DEVICES=0-7 bash _serve_tp8_mtp.sh --port 8000 --log /tmp/single_tp8.log
#
# 关键参数:
#   --port <port>       vLLM API 端口 (default 8001)
#   --log <logfile>     日志路径 (default /tmp/_serve_tp8_mtp.log)
#   --mtp <0|1>         MTP 开关 (default 1)
#   --prefill-or-decode 是否为 PD prefill 实例 (传 "prefill" 关闭 MTP)

set -euo pipefail

PORT=8001
LOG=/tmp/_serve_tp8_mtp.log
MTP=1
ROLE="decode"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --log)  LOG="$2";  shift 2 ;;
    --mtp)  MTP="$2";  shift 2 ;;
    --prefill-or-decode) ROLE="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

# PD prefill 实例不启用 MTP
if [ "$ROLE" = "prefill" ]; then
  MTP=0
fi

MODEL_PATH=/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-npu16

echo "=== $(date) : starting TP8 server ==="
echo "  port=$PORT mtp=$MTP role=$ROLE log=$LOG model=$(basename $MODEL_PATH)"

mkdir -p "$(dirname "$LOG")"

python3 perf_opt/scripts/start_server_phase5.py \
  --port "$PORT" \
  --mtp "$MTP" \
  --static-kernel true \
  --kv-cache-dtype fp8 \
  --cpu-binding true \
  --dsa-cp true \
  --multistream-overlap true \
  "$MODEL_PATH" \
  > "$LOG" 2>&1 &

echo "pid=$! (waiting for startup log...)"

# 等待 vLLM 就绪
for i in $(seq 1 60); do
  sleep 10
  if grep -q "Uvicorn running on" "$LOG" 2>/dev/null; then
    echo "=== server ready on port $PORT ==="
    exit 0
  fi
  echo "  ... $((i*10))s (MTP=$MTP)"
done

echo "ERROR: startup timeout after 600s"
exit 1
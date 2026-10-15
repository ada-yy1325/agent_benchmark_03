#!/usr/bin/env bash
# dp2tp8_bench.sh — 并行跑 DP2×TP8 四个并发档的全部 benchmark
# 前提：两个 VLLM TP8 服务器已在线 (port 8000, 8001)
# 用法：bash dp2tp8_bench.sh

set -euo pipefail

ROOT=/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test
BENCH=$ROOT/perf_opt/scripts/bench_serving.py
MODEL_DIR=$ROOT/models/DeepSeek-V4-Flash-w8a8-self
OUT_DIR=$ROOT/perf_opt/results/topo/dp2tp8
SERVED=dsv4
PY=/opt/mamba/bin/python3
TOPO=dp2tp8
INP=4096
OUT=1024
NREQ=128   # total, 每个 server 64

CONCS=(1 8 32 64)
PORTS=(8000 8001)

mkdir -p "$OUT_DIR"

T_START=$(date +%s)

ALL_JOBS=""
for c in "${CONCS[@]}"; do
  for port in "${PORTS[@]}"; do
    label="${TOPO}_port${port}_c${c}_${INP}k"
    json="$OUT_DIR/${label}.json"
    npu_csv="$OUT_DIR/${label}_npu.csv"
    log="$OUT_DIR/${label}_log.txt"
    sess="bench_c${c}_p${port}"
    tmux kill-session -t "$sess" 2>/dev/null || true
    tmux new-session -d -s "$sess" \
      "$PY $BENCH \
        --served-name $SERVED \
        --model-dir $MODEL_DIR \
        --base-url http://127.0.0.1:${port}/v1 \
        --input-len $INP \
        --output-len $OUT \
        --concurrency $c \
        --num-requests $((NREQ / 2)) \
        --out $json \
        --case-id $label \
        --npu-stats $npu_csv \
        --temp 0.0 --seed 1 \
        2>&1 | tee $log"
    ALL_JOBS="$ALL_JOBS $label:$json:$sess"
    echo "[$(date +%H:%M:%S)] LAUNCH tmux:$sess c=$c port=$port"
  done
done

echo ""
echo "[$(date +%H:%M:%S)] 全部 8 个 benchmark 已启动，轮询等待..."
echo ""

# 轮询等待所有 JSON 文件出现
declare -A DONE
while true; do
  all_done=true
  for job in $ALL_JOBS; do
    IFS=: read -r label json sess <<< "$job"
    if [[ "${DONE[$label]:-0}" == "1" ]]; then continue; fi
    if [[ -f "$json" ]]; then
      DONE[$label]=1
      ELAPSED=$(( $(date +%s) - T_START ))
      echo "[$(date +%H:%M:%S)] ✅ $label done (elapsed ${ELAPSED}s)"
    else
      all_done=false
    fi
  done
  if $all_done; then break; fi
  sleep 15
done

TOTAL_TIME=$(( $(date +%s) - T_START ))
echo ""
echo "[$(date +%H:%M:%S)] 全部完成，总耗时 ${TOTAL_TIME}s ($((TOTAL_TIME/60))min)"
echo ""

# 用 Python 汇总结果
$PY << 'PYEOF'
import json, os, time
from pathlib import Path

OUT_DIR = Path("/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/perf_opt/results/topo/dp2tp8")
TOPO = "dp2tp8"
INP = 4096
OUT = 1024
CONCS = [1, 8, 32, 64]
PORTS = [8000, 8001]

all_results = []
for c in CONCS:
    port_results = []
    for port in PORTS:
        label = f"{TOPO}_port{port}_c{c}_{INP}k"
        json_path = OUT_DIR / f"{label}.json"
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text())
                port_results.append({
                    "port": port,
                    "ttft_p50": data["ttft_ms"]["p50"],
                    "ttft_p99": data["ttft_ms"]["p99"],
                    "tpot_p50": data["tpot_ms"]["p50"],
                    "tp": data["output_tokens_per_s"],
                    "ok": data["successful_requests"],
                    "total": data["planned_requests"],
                    "success": True,
                })
            except Exception as e:
                port_results.append({"port": port, "success": False, "error": str(e)})
        else:
            port_results.append({"port": port, "success": False, "error": "json not found"})

    ok = [r for r in port_results if r.get("success")]
    if ok:
        total_tp = sum(r["tp"] for r in ok)
        avg_ttft_p50 = sum(r["ttft_p50"] for r in ok) / len(ok)
        avg_ttft_p99 = sum(r["ttft_p99"] for r in ok) / len(ok)
        avg_tpot = sum(r["tpot_p50"] for r in ok) / len(ok)
        total_ok = sum(r["ok"] for r in ok)
        total_req = sum(r["total"] for r in ok)
        r = {
            "concurrency": c,
            "ttft_p50": avg_ttft_p50,
            "ttft_p99": avg_ttft_p99,
            "tpot_p50": avg_tpot,
            "tp": total_tp,
            "ok": total_ok,
            "total": total_req,
            "success": True,
            "detail": port_results,
        }
        print(f"  [c={c}] TTFT p50={avg_ttft_p50:.0f}ms p99={avg_ttft_p99:.0f}ms | TP={total_tp:.1f} tok/s | TPOT={avg_tpot:.1f}ms | ok={total_ok}/{total_req}")
    else:
        r = {"concurrency": c, "success": False, "error": "all ports failed", "detail": port_results}
        print(f"  [c={c}] FAIL")
    all_results.append(r)

summary = {
    "topo": TOPO, "tp_per_server": 8, "num_servers": 2,
    "batched": 8192, "max_seqs": 64,
    "input_len": INP, "output_len": OUT,
    "results": all_results,
}
(OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"\n✅ 汇总结果已写入 {OUT_DIR / '_summary.json'}")
PYEOF

echo ""
echo "[$(date +%H:%M:%S)] 🎉 全部完成！"
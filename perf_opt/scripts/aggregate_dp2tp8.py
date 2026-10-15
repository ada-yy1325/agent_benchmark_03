#!/usr/bin/env python3
import json
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
print(f"\nSummary -> {OUT_DIR / '_summary.json'}")
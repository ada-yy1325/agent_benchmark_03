#!/usr/bin/env python3
"""2.4 bench-only: run c=32/48/64/128 on running server."""
import subprocess, time, json
from pathlib import Path

BENCH = Path(__file__).resolve().parent / "bench_serving.py"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "sched" / "final"

CONCURRENCIES = [32, 48, 64, 128]
INP, OUT, NREQ = 4096, 4096, 128

def run(conc):
    label = f"final_c{conc}_{INP}k"
    out_json = OUT_DIR / f"{label}.json"
    npu_csv = OUT_DIR / f"{label}_npu.csv"
    cmd = [
        "/opt/mamba/bin/python3", str(BENCH),
        "--model", "dsv4",
        "--base-url", "http://127.0.0.1:8000/v1",
        "--inp", str(INP), "--out", str(OUT),
        "--concurrency", str(conc),
        "--num-prompts", str(NREQ),
        "--out", str(out_json),
        "--case-id", label,
        "--npu-stats", str(npu_csv),
        "--temp", "0.0", "--seed", "1",
    ]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    wall = time.time() - t0
    if r.returncode != 0:
        return {"c": conc, "success": False, "error": f"rc={r.returncode}"}
    try:
        data = json.loads(out_json.read_text())
        return {
            "c": conc,
            "ttft_p50": data["ttft_ms"]["p50"],
            "tpot_p50": data["tpot_ms"]["p50"],
            "tp": data["output_tokens_per_s"],
            "ok": data["successful_requests"],
            "total": data["planned_requests"],
            "wall_s": round(wall),
            "success": True,
        }
    except Exception as e:
        return {"c": conc, "success": False, "error": str(e)}

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for c in CONCURRENCIES:
        print(f"\n>> c={c}", flush=True)
        r = run(c)
        results.append(r)
        if r.get("success"):
            print(f"  TTFT={r['ttft_p50']:.0f}ms TP={r['tp']:.1f} TPOT={r['tpot_p50']:.1f}ms ok={r['ok']}/{r['total']}", flush=True)
        else:
            print(f"  FAIL: {r.get('error','')}", flush=True)
    
    with open(OUT_DIR / "_summary.json", "w") as f:
        json.dump({"batched": 8192, "max_seqs": 64, "results": results}, f, indent=2)
    
    print(f"\n{'='*50}\n  FINAL (batched=8192, max-seqs=64)", flush=True)
    print(f"  {'c':>4} | {'TTFT':>10} | {'TP':>8} | {'TPOT':>8}", flush=True)
    for r in results:
        if r.get("success"):
            print(f"  {r['c']:>4} | {r['ttft_p50']:.0f}ms{'':>4} | {r['tp']:.1f}{'':>4} | {r['tpot_p50']:.1f}ms", flush=True)
    print("  DONE.", flush=True)

if __name__ == "__main__":
    main()
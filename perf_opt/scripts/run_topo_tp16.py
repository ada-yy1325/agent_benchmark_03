#!/usr/bin/env python3
"""Phase 3: TP16 topology benchmark."""
import subprocess, time, json, os, sys
from pathlib import Path

TOPO = "tp16"
TP, DP = 16, 1
BENCH = Path(__file__).resolve().parent / "bench_serving.py"
BASE_URL = "http://127.0.0.1:8000/v1"
SERVED = "dsv4"
MODEL_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-self"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "topo" / TOPO
HEALTH_URL = "http://127.0.0.1:8000/health"
WORKDIR = str(Path(__file__).resolve().parent.parent)
START_SCRIPT = str(Path(__file__).resolve().parent / "start_server_topo.py")

import urllib.request

CONCURRENCIES = [1, 8, 32, 64]
INP = 4096
OUT = 1024
NREQ = 128
BATCHED = 8192
MAX_SEQS = 64


def kill_server():
    print("  [KILL] Stopping old vLLM...", flush=True)
    os.system("pkill -9 -f VLLM 2>/dev/null")
    time.sleep(3)
    os.system("fuser -k 8000/tcp 2>/dev/null")
    time.sleep(2)


def start_server():
    name = f"topo_{TOPO}"
    os.system(f"tmux kill-session -t {name} 2>/dev/null")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUT_DIR / f"_server_{TOPO}.log"
    wrapper = OUT_DIR / "_start.sh"
    with open(wrapper, "w") as f:
        f.write(f"""#!/bin/bash
set -e
export PATH=/opt/mamba/bin:$PATH
cd {WORKDIR}
exec python3 {START_SCRIPT} --tp {TP} --dp {DP} --batched {BATCHED} --max-seqs {MAX_SEQS} --port 8000 2>&1 | tee {log_file}
""")
    os.chmod(wrapper, 0o755)
    os.system(f"tmux new-session -d -s {name} bash {wrapper}")
    print(f"  [START] tmux:{name} TP={TP} DP={DP} (world={TP*DP})", flush=True)
    time.sleep(5)


def wait_healthy(timeout=600):
    print(f"  [WAIT] /health (timeout={timeout}s)...", flush=True)
    for i in range(1, timeout + 1):
        try:
            urllib.request.urlopen(HEALTH_URL, timeout=3)
            print(f"  [OK] Healthy after {i}s", flush=True)
            return True
        except Exception:
            if i % 30 == 0:
                print(f"  [WAIT] {i}s...", flush=True)
            time.sleep(1)
    print("  [FAIL] Health timeout", flush=True)
    return False


def run_bench(conc):
    label = f"{TOPO}_c{conc}_{INP}k"
    out_json = OUT_DIR / f"{label}.json"
    npu_csv = OUT_DIR / f"{label}_npu.csv"
    cmd = [
        "/opt/mamba/bin/python3", str(BENCH),
        "--served-name", SERVED,
        "--model-dir", MODEL_DIR,
        "--base-url", BASE_URL,
        "--input-len", str(INP),
        "--output-len", str(OUT),
        "--concurrency", str(conc),
        "--num-requests", str(NREQ),
        "--out", str(out_json),
        "--case-id", label,
        "--npu-stats", str(npu_csv),
        "--temp", "0.0", "--seed", "1",
    ]
    t0 = time.time()
    print(f"  [RUN] c={conc}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    wall = time.time() - t0
    if r.returncode != 0:
        err_file = OUT_DIR / f"{label}_err.txt"
        err_file.write_text(r.stderr)
        return {"concurrency": conc, "success": False, "error": f"rc={r.returncode}", "wall_s": round(wall)}
    try:
        data = json.loads(out_json.read_text())
        return {
            "concurrency": conc,
            "ttft_p50": data["ttft_ms"]["p50"],
            "ttft_p99": data["ttft_ms"]["p99"],
            "tpot_p50": data["tpot_ms"]["p50"],
            "tp": data["output_tokens_per_s"],
            "ok": data["successful_requests"],
            "total": data["planned_requests"],
            "wall_s": round(wall),
            "success": True,
        }
    except Exception as e:
        return {"concurrency": conc, "success": False, "error": str(e), "wall_s": round(wall)}


def main():
    print(f"\n{'='*60}")
    print(f"  Phase 3 · {TOPO.upper()} · TP={TP} DP={DP} WORLD={TP*DP}")
    print(f"  batched={BATCHED}, max-seqs={MAX_SEQS}, input={INP}, output={OUT}")
    print(f"{'='*60}\n", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kill_server()

    print(f"  [STARTUP] Launching {TOPO.upper()} server...", flush=True)
    t_start = time.time()
    start_server()

    if not wait_healthy(600):
        print("  [ABORT] Server failed to start", flush=True)
        return

    load_time = time.time() - t_start
    print(f"  [INFO] Model loaded in {load_time:.0f}s", flush=True)

    all_results = []
    for idx, c in enumerate(CONCURRENCIES):
        print(f"\n  {chr(0x2500)*50}")
        print(f"  >> c={c} ({idx+1}/{len(CONCURRENCIES)})", flush=True)
        r = run_bench(c)
        all_results.append(r)
        if r.get("success"):
            print(f"  [c{c}] TTFT p50={r['ttft_p50']:.0f}ms p99={r['ttft_p99']:.0f}ms | "
                  f"TP={r['tp']:.1f} tok/s | TPOT={r['tpot_p50']:.1f}ms | "
                  f"ok={r['ok']}/{r['total']} | wall={r['wall_s']}s", flush=True)
        else:
            print(f"  [c{c}] FAIL: {r.get('error','')}", flush=True)

    summary = {
        "topo": TOPO, "tp": TP, "dp": DP, "world": TP * DP,
        "batched": BATCHED, "max_seqs": MAX_SEQS,
        "input_len": INP, "output_len": OUT,
        "load_time_s": int(load_time),
        "results": all_results,
    }
    with open(OUT_DIR / "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  [SUMMARY] -> {OUT_DIR / '_summary.json'}", flush=True)

    print(f"\n{'='*65}")
    print(f"  {TOPO.upper()} RESULTS  (TP={TP} DP={DP})")
    print(f"  {'c':>4} | {'TTFT p50':>12} | {'TTFT p99':>12} | {'TP tok/s':>10} | {'TPOT':>8} | {'ok':>5}")
    print(f"  {chr(0x2500)*63}")
    for r in all_results:
        c = r["concurrency"]
        if r.get("success"):
            print(f"  {c:>4} | {r['ttft_p50']:>9.0f} ms | {r['ttft_p99']:>9.0f} ms | "
                  f"{r['tp']:>9.1f}  | {r['tpot_p50']:>5.1f} ms | {r['ok']:>3}/{r['total']}")
        else:
            print(f"  {c:>4} | {'FAIL':>12} | {'FAIL':>12} | {'FAIL':>10} | {'FAIL':>8} | 0/0")
    print(f"  {'='*65}")
    print(f"  DONE. Server still running in tmux:topo_{TOPO}", flush=True)


if __name__ == "__main__":
    main()

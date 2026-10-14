#!/usr/bin/env python3
"""
Phase 2.4: Combination verification.
Start one server with batched=8192, max-seqs=64.
Run c=32, 48, 64, 128 on same instance.
"""
import subprocess, time, json, os
from pathlib import Path

BENCH = Path(__file__).resolve().parent / "bench_serving.py"
BASE_URL = "http://127.0.0.1:8000/v1"
SERVED = "dsv4"
MODEL_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-self"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "sched" / "final"
HEALTH_URL = "http://127.0.0.1:8000/health"
WORKDIR = str(Path(__file__).resolve().parent.parent)
START_SCRIPT = str(Path(__file__).resolve().parent / "start_server_param.py")

import urllib.request

CONCURRENCIES = [32, 48, 64, 128]
INP = 4096
OUT = 4096
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
    name = "p24"
    os.system(f"tmux kill-session -t {name} 2>/dev/null")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wrapper = OUT_DIR / "_start_p24.sh"
    with open(wrapper, "w") as f:
        f.write(f"#!/bin/bash\nexport PATH=/opt/mamba/bin:$PATH\ncd {WORKDIR}\nexec python3 {START_SCRIPT} --batched {BATCHED} --port 8000 --max-seqs {MAX_SEQS}\n")
    os.chmod(wrapper, 0o755)
    os.system(f"tmux new-session -d -s {name} bash {wrapper}")
    print(f"  [START] tmux:{name} batched={BATCHED} max-seqs={MAX_SEQS}", flush=True)
    time.sleep(3)

def wait_healthy(timeout=480):
    for i in range(timeout):
        try:
            urllib.request.urlopen(HEALTH_URL, timeout=3)
            print(f"  [OK] Healthy after {i+1}s", flush=True)
            return True
        except Exception:
            time.sleep(1)
    print("  [FAIL] Health check timeout", flush=True)
    return False

def run_bench(conc):
    label = f"final_c{conc}_{INP}k"
    out_json = OUT_DIR / f"{label}.json"
    npu_csv = OUT_DIR / f"{label}_npu.csv"
    case_id = label
    cmd = [
        "/opt/mamba/bin/python3", str(BENCH),
        "--model", SERVED,
        "--base-url", BASE_URL,
        "--inp", str(INP),
        "--out", str(OUT),
        "--concurrency", str(conc),
        "--num-prompts", str(NREQ),
        "--out", str(out_json),
        "--case-id", case_id,
        "--npu-stats", str(npu_csv),
        "--temp", "0.0",
        "--seed", "1",
    ]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    wall = time.time() - t0
    if r.returncode != 0:
        return {"concurrency": conc, "success": False, "error": f"rc={r.returncode}", "stderr": r.stderr[-200:]}
    try:
        data = json.loads(out_json.read_text())
        return {
            "concurrency": conc,
            "ttft_p50": data["ttft_ms"]["p50"],
            "tpot_p50": data["tpot_ms"]["p50"],
            "tp": data["output_tokens_per_s"],
            "ok": data["successful_requests"],
            "total": data["planned_requests"],
            "wall_s": round(wall),
            "success": True,
        }
    except Exception as e:
        return {"concurrency": conc, "success": False, "error": str(e)}

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kill_server()
    start_server()
    if not wait_healthy(480):
        print("Server failed to start", flush=True)
        return
    
    all_results = []
    for c in CONCURRENCIES:
        print(f"\n  >> c={c}", flush=True)
        r = run_bench(c)
        all_results.append(r)
        if r.get("success"):
            print(f"  [c{c}] TTFT={r['ttft_p50']:.0f}ms TP={r['tp']:.1f} ok={r['ok']}/{r['total']}", flush=True)
        else:
            print(f"  [c{c}] FAIL: {r.get('error','')}", flush=True)
    
    with open(OUT_DIR / "_summary.json", "w") as f:
        json.dump({"batched": BATCHED, "max_seqs": MAX_SEQS, "results": all_results}, f, indent=2)
    
    print(f"\n{'='*50}\n  FINAL TABLE", flush=True)
    hdr = f"  {'c':>4} | {'TTFT':>10} | {'TP':>8} | {'TPOT':>8} | ok"
    print(hdr)
    print(f"  {'-'*45}")
    for r in all_results:
        c = r["concurrency"]
        if r.get("success"):
            print(f"  {c:>4} | {r['ttft_p50']:.0f}ms{'':>4} | {r['tp']:.1f}{'':>4} | {r['tpot_p50']:.1f}ms{'':>2} | {r['ok']}/{r['total']}")
        else:
            print(f"  {c:>4} | FAIL")
    print("  DONE.")

if __name__ == "__main__":
    main()
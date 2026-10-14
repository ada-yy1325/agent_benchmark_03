#!/usr/bin/env python3
"""
Phase 2.2: Scan --max-num-batched-tokens.
For each value: restart server, run c32_4k + c64_4k, save results.
"""
import subprocess, time, json, os, urllib.request
from pathlib import Path

BENCH = Path(__file__).resolve().parent / "bench_serving.py"
BASE_URL = "http://127.0.0.1:8000/v1"
SERVED = "dsv4"
MODEL_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-self"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "sched" / "bat_scan"
HEALTH_URL = "http://127.0.0.1:8000/health"

CONTROL_POINTS = [
    ("c32_4k", 4096, 1024, 32, 128),
    ("c64_4k", 4096, 1024, 64, 128),
]

def health_ok():
    try:
        req = urllib.request.urlopen(HEALTH_URL, timeout=5)
        return req.status == 200
    except:
        return False

def wait_healthy(max_wait=600):
    start = time.time()
    while time.time() - start < max_wait:
        if health_ok():
            print(f"  [OK] Healthy after {time.time()-start:.0f}s", flush=True)
            return True
        time.sleep(10)
    return False

def kill_server():
    print("  [STOP] Killing old vLLM...", flush=True)
    # Kill VLLM processes by their actual process names on this platform
    os.system("pkill -9 -f 'VLLM::EngineC' 2>/dev/null")
    os.system("pkill -9 -f 'VLLM::Worke' 2>/dev/null")
    # Also kill any python process serving on port 8000
    os.system("fuser -k 8000/tcp 2>/dev/null")
    time.sleep(5)
    os.system("fuser -k 8000/tcp 2>/dev/null")
    time.sleep(3)
    # Verify port is free
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = s.connect_ex(('127.0.0.1', 8000))
    s.close()
    if result == 0:
        print("  [WARN] Port 8000 still occupied! Retrying pkill...", flush=True)
        os.system("pkill -9 -f 'VLLM' 2>/dev/null")
        time.sleep(5)
        os.system("fuser -k 8000/tcp 2>/dev/null")
        time.sleep(3)

def start_server(batched_tokens):
    name = f"sched_bat{batched_tokens}"
    os.system(f"tmux kill-session -t {name} 2>/dev/null")
    workdir = str(Path(__file__).resolve().parent.parent)
    script = str(Path(__file__).resolve().parent / "start_server_param.py")
    # Use bash -c to handle the long command safely
    inner = f"cd {workdir} && /opt/mamba/bin/python3 {script} --batched {batched_tokens} --port 8000 --max-seqs 32"
    cmd = f"tmux new-session -d -s {name} bash -c '{inner} 2>&1'"
    print(f"  [START] tmux:{name} batched={batched_tokens}", flush=True)
    os.system(cmd)
    time.sleep(5)

def run_bench(batched, cid, inp, out, conc, nreq):
    label = f"bat{batched}_{cid}"
    out_json = OUT_DIR / f"{label}.json"
    cmd = [
        "/opt/mamba/bin/python3", str(BENCH),
        "--base-url", BASE_URL, "--served-name", SERVED,
        "--model-dir", MODEL_DIR,
        "--input-len", str(inp), "--output-len", str(out),
        "--concurrency", str(conc), "--num-requests", str(nreq),
        "--result-file", str(out_json),
        "--seed", "42", "--temperature", "0", "--ignore-eos", "true",
    ]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    wall = time.time() - t0
    if r.returncode == 0 and out_json.exists():
        with open(out_json) as f:
            data = json.load(f)
        try:
            s = data["benchmark"]["bench_stats"]
            return {
                "batched": batched, "case": cid,
                "ttft_p50": s["ttft_p50_ms"], "tpot_p50": s["tpot_p50_ms"],
                "tp": s["output_tokens_per_second"],
                "ok": data["benchmark"]["num_completed"],
                "total": data["benchmark"]["num_total"],
                "wall_s": round(wall), "success": True,
            }
        except KeyError as e:
            return {"batched": batched, "case": cid, "success": False, "error": str(e)}
    return {"batched": batched, "case": cid, "success": False, "error": f"rc={r.returncode}"}

def main():
    values = [2048, 4096, 8192, 16384]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    
    for bv in values:
        print(f"\n{'='*50}\n>> batched={bv}\n{'='*50}", flush=True)
        kill_server()
        start_server(bv)
        if not wait_healthy(480):
            print(f"  [SKIP] Server failed", flush=True)
            continue
        res = {}
        for cid, inp, out, conc, nreq in CONTROL_POINTS:
            r = run_bench(bv, cid, inp, out, conc, nreq)
            res[f"bat{bv}_{cid}"] = r
            if r.get("success"):
                print(f"  [{cid}] TTFT={r['ttft_p50']:.0f}ms TP={r['tp']:.1f} ok={r['ok']}/{r['total']}", flush=True)
            else:
                print(f"  [{cid}] FAIL", flush=True)
        all_results.append({"batched_tokens": bv, "results": res})
    
    with open(OUT_DIR / "_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*50}\n  SUMMARY TABLE", flush=True)
    hdr = f"  {'batched':>8} | {'c32 TTFT':>10} | {'c32 TP':>8} | {'c64 TTFT':>10} | {'c64 TP':>8}"
    print(hdr)
    print(f"  {'-'*55}")
    for e in all_results:
        b = e["batched_tokens"]
        c32 = e["results"].get(f"bat{b}_c32_4k", {})
        c64 = e["results"].get(f"bat{b}_c64_4k", {})
        c32t = f"{c32['ttft_p50']:.0f}ms" if c32.get('success') else "FAIL"
        c32p = f"{c32['tp']:.1f}" if c32.get('success') else "-"
        c64t = f"{c64['ttft_p50']:.0f}ms" if c64.get('success') else "FAIL"
        c64p = f"{c64['tp']:.1f}" if c64.get('success') else "-"
        print(f"  {b:>8} | {c32t:>10} | {c32p:>8} | {c64t:>10} | {c64p:>8}")
    print("  DONE.")

if __name__ == "__main__":
    main()
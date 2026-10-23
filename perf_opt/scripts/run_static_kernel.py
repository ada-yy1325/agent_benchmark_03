#!/usr/bin/env python3
"""Mini runner — static_kernel_on benchmarks only (4K, c1/c8/c32/c64)."""
import json, os, sys, subprocess, time
from datetime import datetime

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPTS_DIR))
PYTHON = sys.executable
BENCH = os.path.join(SCRIPTS_DIR, "bench_serving.py")
MODEL_DIR = os.path.join(REPO_ROOT, "models", "DeepSeek-V4-Flash-w8a8-self")
OUT_DIR = "/tmp/phase5/static_kernel_on"
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = "/tmp/phase5/_session_5.3c.log"
PORT = 8000

CASES = [
    ("static_kernel_on_c1_4k",   4096, 1024, 1,   16),
    ("static_kernel_on_c8_4k",   4096, 1024, 8,   64),
    ("static_kernel_on_c32_4k",  4096, 1024, 32, 128),
    ("static_kernel_on_c64_4k",  4096, 1024, 64, 128),
]

def tstamp():
    return datetime.now().strftime("%H:%M:%S")

def log(msg):
    line = f"[static_kernel] {tstamp()} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

def point_ok(out_json):
    try:
        d = json.load(open(out_json))
        if d["successful_requests"] != d["planned_requests"]:
            return False
        exp = d["total_output_tokens"] / d["output_tokens_per_s"]
        drift = abs(d["wall_time_s"] - exp) / max(d["wall_time_s"], exp, 1) * 100
        return drift <= 15.0
    except Exception:
        return False

def run_point(label, inp, out, conc, nreq):
    out_json = os.path.join(OUT_DIR, f"{label}.json")
    npu_csv = os.path.join(OUT_DIR, f"{label}_npu.csv")
    bench_log = os.path.join(OUT_DIR, f"{label}_bench.log")

    # Skip if already done and self-consistent
    if os.path.isfile(out_json) and point_ok(out_json):
        log(f"SKIP {label} (exists, consistent)")
        return

    cmd = [PYTHON, BENCH,
           "--served-name", "dsv4", "--model-dir", MODEL_DIR,
           "--base-url", f"http://127.0.0.1:{PORT}/v1",
           "--input-len", str(inp), "--output-len", str(out),
           "--concurrency", str(conc), "--num-requests", str(nreq),
           "--out", out_json, "--case-id", label, "--npu-stats", npu_csv,
           "--temp", "0.0", "--seed", "42"]

    t0 = time.time()
    with open(bench_log, "w") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    wall = time.time() - t0

    if result.returncode != 0:
        log(f"ERROR {label}: bench exit {result.returncode}")
        return

    if not point_ok(out_json):
        log(f"WARN {label}: result not self-consistent")
        return

    d = json.load(open(out_json))
    log(f"{label}: TTFT={d['ttft_ms']['p50']:.0f}ms TPOT={d['tpot_ms']['p50']:.1f} "
        f"tok/s={d['output_tokens_per_s']:.1f} wall={wall:.0f}s "
        f"drift={abs(d['wall_time_s'] - d['total_output_tokens']/d['output_tokens_per_s']) / max(d['wall_time_s'], d['total_output_tokens']/d['output_tokens_per_s'], 1) * 100:.1f}% OK")

def main():
    log("START static_kernel_on sweep")
    for label, inp, out, conc, nreq in CASES:
        run_point(label, inp, out, conc, nreq)
    log("DONE static_kernel_on sweep")

if __name__ == "__main__":
    main()
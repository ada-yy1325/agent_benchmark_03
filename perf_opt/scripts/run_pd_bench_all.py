#!/usr/bin/env python3
"""Run P/D benchmarks from remote — loops safe even if SSH drops."""
import subprocess, sys, os, time

OUT = "/tmp/pd_bench"
ROOT = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test"
BASE = [
    sys.executable, "perf_opt/scripts/bench_serving.py",
    "--served-name", "dsv4",
    "--model-dir", f"{ROOT}/models/DeepSeek-V4-Flash-w8a8-npu16",
    "--base-url", "http://127.0.0.1:8000/v1",
    "--temp", "0.0", "--seed", "42",
]


def run(input_len, output_len, concurrency, case_id, npu_csv):
    out_path = os.path.join(OUT, f"{case_id}.json")
    if os.path.exists(out_path):
        print(f"SKIP {case_id} — already exists")
        return True
    cmd = BASE + [
        "--input-len", str(input_len),
        "--output-len", str(output_len),
        "--concurrency", str(concurrency),
        "--num-requests", "16",
        "--out", out_path,
        "--case-id", case_id,
        "--npu-stats", os.path.join(OUT, npu_csv),
    ]
    print(f"RUN {case_id}  (IL={input_len} OL={output_len} c={concurrency})")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    print(f"DONE {case_id} rc={r.returncode}  {elapsed:.0f}s")
    return r.returncode == 0


if __name__ == "__main__":
    # No-MTP 4K/1K c=1,4,8,16
    for c in [1, 4, 8, 16]:
        if not run(4096, 1024, c, f"pd_nomtp_4k1k_c{c}", f"npu_nomtp_4k1k_c{c}.csv"):
            print(f"FAIL at NoMTP 4K/1K c={c}")
            sys.exit(1)
    print("ALL NoMTP 4K/1K DONE")
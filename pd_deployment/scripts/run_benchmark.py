#!/usr/bin/env python3
"""
Universal P/D Benchmark Runner — runs a full benchmark matrix.
Usage:
  python3 pd_deployment/scripts/run_benchmark.py [--mtp] [--quick]

  --mtp:    Test MTP ON P/D
  --quick:  Quick smoke test (1 concurrency only, 4 requests)
"""

import argparse, json, os, subprocess, sys, time

REPO = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test"
MODEL = f"{REPO}/models/DeepSeek-V4-Flash-w8a8-npu16"
OUT = "/tmp/pd_bench"

# Benchmark matrices
MATRIX_QUICK = [(4096, 128, [1])]                     # 4K → 128, c=1
MATRIX_FULL = [
    (4096, 128, [1, 4, 8, 16]),                       # 4K → 128
    (16384, 128, [1, 4, 8, 16]),                      # 16K → 128
    (4096, 1024, [1, 4, 8, 16]),                      # 4K → 1K (core)
]


def run_bench(input_len, output_len, concurrency, case_id, npu_csv):
    out_path = f"{OUT}/{case_id}.json"
    if os.path.exists(out_path):
        print(f"  SKIP {case_id} — exists")
        return True
    cmd = [
        sys.executable, f"{REPO}/perf_opt/scripts/bench_serving.py",
        "--served-name", "dsv4",
        "--model-dir", MODEL,
        "--base-url", "http://127.0.0.1:8000/v1",
        "--input-len", str(input_len),
        "--output-len", str(output_len),
        "--concurrency", str(concurrency),
        "--num-requests", "16",
        "--out", out_path,
        "--case-id", case_id,
        "--npu-stats", f"{OUT}/{npu_csv}",
        "--temp", "0.0", "--seed", "42",
    ]
    t0 = time.time()
    r = subprocess.run(cmd)
    elapsed = time.time() - t0
    ok = r.returncode == 0
    print(f"  {'✅' if ok else '❌'} {case_id}  {elapsed:.0f}s")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    tag = "mtp" if args.mtp else "nomtp"
    matrix = MATRIX_QUICK if args.quick else MATRIX_FULL

    print(f"\n{'═'*60}")
    print(f"  P/D Benchmark — {'MTP ON' if args.mtp else 'No-MTP'}  {'(quick)' if args.quick else '(full)'}")
    print(f"{'═'*60}\n")

    ok = 0
    fail = 0
    for il, ol, concurrency_list in matrix:
        for c in concurrency_list:
            case_id = f"pd_{tag}_{il // 1024}k{ol}_c{c}"
            npu_csv = f"npu_{tag}_{il // 1024}k{ol}_c{c}.csv"
            if run_bench(il, ol, c, case_id, npu_csv):
                ok += 1
            else:
                fail += 1

    print(f"\n{'═'*60}")
    print(f"  Done: {ok} passed, {fail} failed")
    print(f"  Results: {OUT}/pd_{tag}_*.json")
    print(f"  Read: python3 {REPO}/perf_opt/scripts/pd_read_results.py")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
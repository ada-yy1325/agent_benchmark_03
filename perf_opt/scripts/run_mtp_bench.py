#!/usr/bin/env python3
"""Phase 4: MTP benchmark runner.

Usage (on remote, server must already be running):
    python3 perf_opt/scripts/run_mtp_bench.py --port 8000 --mtp 1 --output perf_opt/results/mtp/mtp_1
    python3 perf_opt/scripts/run_mtp_bench.py --port 8000 --mtp off --output perf_opt/results/mtp/mtp_off
"""
import argparse, json, os, sys, subprocess, time, re
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_SCRIPT = os.path.join(SCRIPT_DIR, "bench_serving.py")
MODEL_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-self"

INPUT_LEN = 4096
OUTPUT_LEN = 1024
CONCURRENCIES = [1, 8, 32]
NUM_REQUESTS = {1: 32, 8: 64, 32: 128}

def get_metrics(port):
    """Query vLLM /metrics endpoint, extract speculation-related metrics."""
    try:
        import urllib.request
        url = f"http://127.0.0.1:{port}/metrics"
        resp = urllib.request.urlopen(url, timeout=5)
        text = resp.read().decode()
        metrics = {}
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            for key in ["vllm:speculation_draft_acceptance_rate",
                        "vllm:num_speculative_tokens_total",
                        "vllm:num_speculative_tokens_accepted_total"]:
                if key in line:
                    parts = line.split()
                    if parts:
                        try:
                            metrics[key] = float(parts[-1])
                        except ValueError:
                            metrics[key] = parts[-1]
        return metrics, None
    except Exception as e:
        return {}, str(e)

def run_bench(port, concurrency, case_id, out_file):
    """Run bench_serving.py for one concurrency point."""
    num_req = NUM_REQUESTS.get(concurrency, 64)
    cmd = [
        sys.executable, BENCH_SCRIPT,
        "--base-url", f"http://127.0.0.1:{port}/v1",
        "--served-name", "dsv4",
        "--model-dir", MODEL_DIR,
        "--input-len", str(INPUT_LEN),
        "--output-len", str(OUTPUT_LEN),
        "--concurrency", str(concurrency),
        "--num-requests", str(num_req),
        "--case-id", case_id,
        "--out", out_file,
        "--npu-stats", out_file.replace(".json", "_npu.csv"),
        "--seed", "42",
        "--temp", "0.0",
    ]
    print(f"  [{case_id}] Running...", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  [{case_id}] FAIL rc={result.returncode}: {result.stderr[-300:]}", flush=True)
        return None, elapsed
    print(f"  [{case_id}] OK ({elapsed:.0f}s)", flush=True)
    try:
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mtp", type=str, required=True,
                        choices=["off", "0", "1", "2", "3"])
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--concurrencies", type=str, default="1,8,32")
    args = parser.parse_args()

    mtp_label = f"mtp_{args.mtp}" if args.mtp not in ("off", "0") else "mtp_off"
    os.makedirs(args.output, exist_ok=True)

    results = {
        "phase": "4-mtp-bench", "date": datetime.now().isoformat(),
        "mtp": args.mtp, "port": args.port,
        "params": {"input_len": INPUT_LEN, "output_len": OUTPUT_LEN,
                   "batched": 8192, "max_seqs": 64},
        "benchmarks": {}, "metrics_before": {}, "metrics_after": {},
    }

    # Metrics before
    print(f"\n[MTP={args.mtp}] Metrics BEFORE on port {args.port}...", flush=True)
    mb, err = get_metrics(args.port)
    results["metrics_before"] = mb
    print(f"  Before: {mb}" + (f" (err: {err})" if err else ""), flush=True)

    # Run benchmarks
    cs = [int(x) for x in args.concurrencies.split(",")]
    for c in cs:
        case_id = f"4k_out1024_c{c}_{mtp_label}"
        out_file = os.path.join(args.output, f"{case_id}.json")
        r, elapsed = run_bench(args.port, c, case_id, out_file)
        if r:
            results["benchmarks"][f"c{c}"] = {
                "ttft_p50_ms": r.get("ttft_p50_ms"),
                "ttft_p99_ms": r.get("ttft_p99_ms"),
                "tpot_p50_ms": r.get("tpot_p50_ms"),
                "tp_tok_s": r.get("tp_tok_s"),
                "ok": r.get("ok"), "total": r.get("total"),
                "elapsed_s": elapsed,
            }

    # Metrics after
    print(f"\n[MTP={args.mtp}] Metrics AFTER...", flush=True)
    ma, err = get_metrics(args.port)
    results["metrics_after"] = ma
    print(f"  After: {ma}" + (f" (err: {err})" if err else ""), flush=True)

    # Acceptance rate
    try:
        bs = float(mb.get("vllm:num_speculative_tokens_accepted_total", 0))
        at_s = float(ma.get("vllm:num_speculative_tokens_accepted_total", 0))
        bt = float(mb.get("vllm:num_speculative_tokens_total", 0))
        at_t = float(ma.get("vllm:num_speculative_tokens_total", 0))
        accepted = at_s - bs
        total = at_t - bt
        if total > 0:
            results["acceptance_rate"] = accepted / total
            print(f"\n  Acceptance: {accepted:.0f}/{total:.0f} = {results['acceptance_rate']:.2%}", flush=True)
        else:
            results["acceptance_rate"] = None
            print(f"\n  Acceptance: N/A (no speculative tokens)", flush=True)
    except Exception as e:
        results["acceptance_rate"] = None
        print(f"\n  Acceptance: error ({e})", flush=True)

    # Save
    summary_file = os.path.join(args.output, f"_summary_{mtp_label}.json")
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSummary → {summary_file}", flush=True)

    # Print table
    print(f"\n{'='*60}")
    print(f"MTP={args.mtp}  Results")
    print(f"{'='*60}")
    for c in cs:
        b = results["benchmarks"].get(f"c{c}", {})
        if b:
            print(f"  c={c:>2}: TTFT={str(b.get('ttft_p50_ms','?')):>8}ms  "
                  f"TPOT={str(b.get('tpot_p50_ms','?')):>6}ms  "
                  f"TP={str(b.get('tp_tok_s','?')):>6} tok/s  "
                  f"OK={b.get('ok')}/{b.get('total')}")
    acc = results.get("acceptance_rate")
    print(f"  Acceptance: {acc:.2%}" if isinstance(acc, float) else f"  Acceptance: {acc}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
        with open(out_file) as f:
            return json.load(f), elapsed
    except Exception as e:
        print(f"  [{case_id}] Read error: {e}", flush=True)
        return None, elapsed
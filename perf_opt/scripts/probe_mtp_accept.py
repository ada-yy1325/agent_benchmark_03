#!/usr/bin/env python3
"""Quick MTP=1 acceptance-rate probe for the running server on port 8000.
Runs c=1 benchmark, captures metrics before/after, computes acceptance rate."""
import subprocess, json, urllib.request, time, os, sys

METRICS_URL = "http://127.0.0.1:8000/metrics"
RESULTS_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/perf_opt/results/mtp/mtp_1"
MODEL_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-self"
BENCH_SCRIPT = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/perf_opt/scripts/bench_serving.py"

os.makedirs(RESULTS_DIR, exist_ok=True)

def get_spec_metrics():
    """Return accepted_total, draft_total from /metrics."""
    try:
        resp = urllib.request.urlopen(METRICS_URL, timeout=10)
        text = resp.read().decode()
        accepted = 0.0
        draft = 0.0
        for line in text.split("\n"):
            if "vllm:spec_decode_num_draft_tokens_total" in line and not line.startswith("#"):
                parts = line.split()
                if parts:
                    try:
                        draft = float(parts[-1])
                    except ValueError:
                        pass
            if "vllm:spec_decode_num_accepted_tokens_total" in line and not line.startswith("#"):
                parts = line.split()
                if parts:
                    try:
                        accepted = float(parts[-1])
                    except ValueError:
                        pass
        return accepted, draft
    except Exception as e:
        return 0.0, 0.0

# 1. Metrics before
print("=== Metrics BEFORE ===")
acc_before, draft_before = get_spec_metrics()
print(f"accepted={acc_before} draft={draft_before}")

# 2. Run benchmark c=1
print("\n=== Running benchmark c=1 (32 req, ~60s) ===")
cmd = [
    sys.executable, BENCH_SCRIPT,
    "--base-url", "http://127.0.0.1:8000/v1",
    "--served-name", "dsv4",
    "--model-dir", MODEL_DIR,
    "--input-len", "4096",
    "--output-len", "1024",
    "--concurrency", "1",
    "--num-requests", "32",
    "--case-id", "mtp1_probe_c1",
    "--out", os.path.join(RESULTS_DIR, "mtp1_probe_c1.json"),
    "--npu-stats", os.path.join(RESULTS_DIR, "mtp1_probe_c1_npu.csv"),
    "--seed", "42",
    "--temp", "0.0",
]
t0 = time.time()
result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
elapsed = time.time() - t0

if result.returncode != 0:
    print(f"BENCH FAILED: {result.stderr[-500:]}")
    sys.exit(1)
print(f"BENCH OK ({elapsed:.0f}s)")

# 3. Read result
with open(os.path.join(RESULTS_DIR, "mtp1_probe_c1.json")) as f:
    bench_result = json.load(f)
print(f"  TTFT p50={bench_result.get('ttft_p50_ms', '?')}ms, "
      f"TPOT p50={bench_result.get('tpot_p50_ms', '?')}ms, "
      f"TP={bench_result.get('tp_tok_s', '?')} tok/s")

# 4. Metrics after
print("\n=== Metrics AFTER ===")
acc_after, draft_after = get_spec_metrics()
print(f"accepted={acc_after} draft={draft_after}")

# 5. Acceptance rate
acc_delta = acc_after - acc_before
draft_delta = draft_after - draft_before
print(f"\n=== Acceptance rate ===")
print(f"Accepted delta: {acc_delta:.0f}")
print(f"Draft delta: {draft_delta:.0f}")
if draft_delta > 0:
    rate = acc_delta / draft_delta
    print(f"Rate: {rate:.2%}")
else:
    rate = None
    print("Rate: N/A (no draft tokens)")

# 6. Save summary
summary = {
    "phase": "4.1-mtp-probe",
    "mtp": 1,
    "bench_result": {
        "ttft_p50_ms": bench_result.get("ttft_p50_ms"),
        "tpot_p50_ms": bench_result.get("tpot_p50_ms"),
        "tp_tok_s": bench_result.get("tp_tok_s"),
    },
    "spec_metrics_before": {"accepted": acc_before, "draft": draft_before},
    "spec_metrics_after": {"accepted": acc_after, "draft": draft_after},
    "acceptance_rate": rate,
    "elapsed_bench_s": elapsed,
}
with open(os.path.join(RESULTS_DIR, "_probe_result.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR}/_probe_result.json")
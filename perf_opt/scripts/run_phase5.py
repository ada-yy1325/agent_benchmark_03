#!/usr/bin/env python3
"""
Phase 5 orchestration: sequential single-variable benchmarks.
Each switch ON/OFF pair tested same-session back-to-back.
Usage (remote):
  setsid nohup python3 perf_opt/scripts/run_phase5.py --step 5.0 > /tmp/phase5/_sess.log 2>&1 &
"""
import os, sys, subprocess, json, time, argparse
from pathlib import Path

WORKDIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test"
os.chdir(WORKDIR)

BENCH = str(Path("perf_opt/scripts/bench_serving.py").resolve())
SERVER = str(Path("perf_opt/scripts/start_server_phase5.py").resolve())
PYTHON = "/opt/mamba/bin/python3"
MODEL_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-self"

# Results to /tmp (shared disk quota full)
RD = Path("/tmp/phase5")
RD.mkdir(parents=True, exist_ok=True)

# Frozen benchmark points: (case_id, input, output, concurrency, num_requests)
CASES = [
    ("c1_4k",  4096, 1024, 1,  16),
    ("c8_4k",  4096, 1024, 8,  64),
    ("c32_4k", 4096, 1024, 32, 128),
    ("c64_4k", 4096, 1024, 64, 128),
]
def log(msg):
    print(f"[phase5] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def pkill():
    subprocess.run(["pkill", "-9", "-f", "VLLM::"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "start_server_phase5"], capture_output=True)
    time.sleep(5)


def wait_health(port, timeout=300):
    for i in range(timeout // 2):
        time.sleep(2)
        try:
            r = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                 f"http://127.0.0.1:{port}/health"],
                capture_output=True, text=True, timeout=10)
            if r.stdout.strip() == "200":
                log(f"Server healthy port={port} (waited {2*(i+1)}s)")
                return True
        except Exception:
            pass
    log(f"ERROR: port={port} not healthy after {timeout}s")
    return False


def start_server(server_args, port=8000, logf=None):
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    cmd = [PYTHON, SERVER, "--port", str(port)] + server_args
    f = open(logf, "w") if logf else subprocess.DEVNULL
    proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    log(f"Started server pid={proc.pid} port={port} args={' '.join(server_args)}")
    return proc


def run_bench(label_prefix, port, out_subdir):
    out_dir = RD / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for case_id, inp, out, conc, nreq in CASES:
        label = f"{label_prefix}_{case_id}"
        out_json = str(out_dir / f"{label}.json")
        npu_csv = str(out_dir / f"{label}_npu.csv")
        cmd = [
            PYTHON, BENCH,
            "--served-name", "dsv4", "--model-dir", MODEL_DIR,
            "--base-url", f"http://127.0.0.1:{port}/v1",
            "--input-len", str(inp), "--output-len", str(out),
            "--concurrency", str(conc), "--num-requests", str(nreq),
            "--out", out_json, "--case-id", label,
            "--npu-stats", npu_csv,
            "--temp", "0.0", "--seed", "42",
        ]
        # Use file redirects to avoid subprocess pipe deadlock (known issue on NPU nodes)
        bench_log = str(out_dir / f"{label}_bench.log")
        t0 = time.time()
        with open(bench_log, "w") as lf:
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
        wall_s = round(time.time() - t0, 1)
        if r.returncode != 0:
            log(f"  FAIL {label}: rc={r.returncode} (see {bench_log})")
            results[case_id] = {"error": f"rc={r.returncode}", "wall_s": wall_s}
            continue
        try:
            data = json.loads(open(out_json).read())
            ttft = data["ttft_ms"]["p50"]
            tpot = data["tpot_ms"]["p50"]
            tok_s = data["output_tokens_per_s"]
            ok = data["successful_requests"]
            total = data["planned_requests"]
            expected = data["total_output_tokens"] / tok_s if tok_s > 0 else 0
            drift = abs(wall_s - expected) / max(wall_s, expected, 1) * 100 if expected > 0 else 0
            self_ok = drift <= 15.0
            results[case_id] = {
                "ttft_p50": ttft, "tpot_p50": tpot, "tok_s": tok_s,
                "ok": ok, "total": total, "wall_s": wall_s,
                "expected_wall_s": round(expected, 1), "drift_pct": round(drift, 2),
                "self_consistent": self_ok,
            }
            flag = "OK" if self_ok else "DRIFT"
            log(f"  {label}: TTFT={ttft:.0f}ms TPOT={tpot:.1f}ms tok/s={tok_s:.1f} "
                f"wall={wall_s}s drift={drift:.1f}% {flag}")
        except Exception as e:
            log(f"  FAIL {label}: {e}")
            results[case_id] = {"error": str(e), "wall_s": wall_s}
    return results
def save_results(name, results, extra=None):
    out = {"name": name, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": results}
    if extra:
        out.update(extra)
    path = RD / f"{name}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    log(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════
# 5.0 BASELINE
# ═══════════════════════════════════════════════════════════
def step_5_0():
    log("=" * 60)
    log("5.0 BASELINE: MTP=OFF, prefix=OFF, cudagraph, all additional=ON")
    pkill()
    start_server(["--mtp", "off"], port=8000,
                 logf=str(RD / "baseline_server.log"))
    if not wait_health(8000):
        return
    results = run_bench("baseline", 8000, "baseline")
    pkill()
    save_results("5_0_baseline", results, {
        "config": {"mtp": "off", "prefix_caching": False, "graph_mode": "cudagraph",
                   "cpu_binding": True, "dsa_cp": True, "multistream_overlap": True,
                   "static_kernel": False}
    })
    return results


# ═══════════════════════════════════════════════════════════
# 5.1 PREFIX CACHING
# ═══════════════════════════════════════════════════════════
def step_5_1():
    log("=" * 60)
    log("5.1 PREFIX CACHING: OFF vs ON (same session)")
    all_r = {}

    # OFF
    pkill()
    start_server(["--mtp", "off"], port=8000,
                 logf=str(RD / "prefix_off_server.log"))
    if not wait_health(8000):
        return
    all_r["prefix_off"] = run_bench("prefix_off", 8000, "prefix_off")
    pkill()

    # ON
    start_server(["--mtp", "off", "--prefix-caching"], port=8000,
                 logf=str(RD / "prefix_on_server.log"))
    if not wait_health(8000):
        return
    all_r["prefix_on"] = run_bench("prefix_on", 8000, "prefix_on")
    pkill()

    save_results("5_1_prefix", all_r)
    return all_r


# ═══════════════════════════════════════════════════════════
# 5.2 GRAPH MODES
# ═══════════════════════════════════════════════════════════
def step_5_2():
    log("=" * 60)
    log("5.2 GRAPH MODES: eager, default, cudagraph, npugraph")
    all_r = {}
    modes = [
        ("eager", ["--mtp", "off", "--graph-mode", "eager"]),
        ("default", ["--mtp", "off", "--graph-mode", "default"]),
        ("cudagraph", ["--mtp", "off", "--graph-mode", "cudagraph"]),
        ("npugraph", ["--mtp", "off", "--graph-mode", "npugraph"]),
    ]
    for mode_name, args in modes:
        log(f"--- graph mode: {mode_name} ---")
        pkill()
        start_server(args, port=8000,
                     logf=str(RD / f"graph_{mode_name}_server.log"))
        if not wait_health(8000):
            all_r[mode_name] = {"error": "server not healthy"}
            continue
        all_r[mode_name] = run_bench(f"graph_{mode_name}", 8000, f"graph_{mode_name}")
        pkill()

    save_results("5_2_graph", all_r)
    return all_r


# ═══════════════════════════════════════════════════════════
# 5.3 ADDITIONAL CONFIG (single-variable OFF vs ON)
# ═══════════════════════════════════════════════════════════
def step_5_3():
    log("=" * 60)
    log("5.3 ADDITIONAL CONFIG: single-variable tests")
    all_r = {}
    # Baseline: cudagraph mode, all additional ON
    base_config = {
        "cpu_binding": "true", "dsa_cp": "true",
        "multistream_overlap": "true", "static_kernel": "false",
    }

    # Test each switch: turn it OFF while others stay ON
    switches = [
        ("cpu_binding_off", {"cpu_binding": "false"}),
        ("dsa_cp_off", {"dsa_cp": "false"}),
        ("multistream_overlap_off", {"multistream_overlap": "false"}),
        ("static_kernel_on", {"static_kernel": "true"}),
    ]

    for name, overrides in switches:
        log(f"--- additional: {name} ---")
        cfg = base_config.copy()
        cfg.update(overrides)
        pkill()
        args = ["--mtp", "off", "--graph-mode", "cudagraph"]
        for k, v in cfg.items():
            args.extend([f"--{k.replace('_', '-')}", v])
        start_server(args, port=8000,
                     logf=str(RD / f"add_{name}_server.log"))
        if not wait_health(8000):
            all_r[name] = {"error": "server not healthy"}
            continue
        all_r[name] = run_bench(f"add_{name}", 8000, f"add_{name}")
        pkill()

    save_results("5_3_add", all_r)
    return all_r


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
STEPS = {
    "5.0": step_5_0,
    "5.1": step_5_1,
    "5.2": step_5_2,
    "5.3": step_5_3,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=str, default="5.0",
                    choices=list(STEPS.keys()),
                    help="Which step to run (default: 5.0)")
    args = ap.parse_args()

    log(f"Phase 5 starting: step={args.step}")
    log(f"Results directory: {RD}")

    try:
        STEPS[args.step]()
        log(f"Step {args.step} COMPLETE.")
    except Exception as e:
        log(f"Step {args.step} FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pkill()
        log("All servers stopped.")
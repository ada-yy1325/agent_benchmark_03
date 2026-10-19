#!/usr/bin/env python3
"""Resume 5.3 additional-config sweep after orchestrator death (Cline session killed it).

Skips any point whose result json exists with full success + wall/throughput
self-consistency; runs the rest sequentially; aggregates into /tmp/phase5/5_3_add.json.
Run on remote:
    setsid nohup python3 /tmp/resume_5_3.py > /tmp/phase5/_session_5.3b.log 2>&1 < /dev/null &
"""
import os, sys, subprocess, json, time
from pathlib import Path

WORKDIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test"
os.chdir(WORKDIR)

BENCH = str(Path("perf_opt/scripts/bench_serving.py").resolve())
SERVER = str(Path("perf_opt/scripts/start_server_phase5.py").resolve())
PYTHON = "/opt/mamba/bin/python3"
MODEL_DIR = WORKDIR + "/models/DeepSeek-V4-Flash-w8a8-self"
RD = Path("/tmp/phase5")

# Frozen benchmark points (same as run_phase5.py)
CASES = [
    ("c1_4k", 4096, 1024, 1, 16),
    ("c8_4k", 4096, 1024, 8, 64),
    ("c32_4k", 4096, 1024, 32, 128),
    ("c64_4k", 4096, 1024, 64, 128),
]


def log(msg):
    print(f"[resume5.3] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def pkill():
    subprocess.run(["pkill", "-9", "-f", "VLLM::"], capture_output=True)
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
                log(f"server healthy ({(i + 1) * 2}s)")
                return True
        except Exception:
            pass
    log("server NOT healthy")
    return False


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


def run_point(label, out_dir, inp, out, conc, nreq, port=8000):
    out_json = str(out_dir / f"{label}.json")
    npu_csv = str(out_dir / f"{label}_npu.csv")
    bench_log = str(out_dir / f"{label}_bench.log")
    cmd = [PYTHON, BENCH,
           "--served-name", "dsv4", "--model-dir", MODEL_DIR,
           "--base-url", f"http://127.0.0.1:{port}/v1",
           "--input-len", str(inp), "--output-len", str(out),
           "--concurrency", str(conc), "--num-requests", str(nreq),
           "--out", out_json, "--case-id", label, "--npu-stats", npu_csv,
           "--temp", "0.0", "--seed", "42"]
    t0 = time.time()
    with open(bench_log, "w") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
    wall = time.time() - t0
    if r.returncode != 0:
        log(f"FAIL {label}: rc={r.returncode}")
        return {"error": f"rc={r.returncode}", "wall_s": round(wall, 1)}
    d = json.load(open(out_json))
    exp = d["total_output_tokens"] / d["output_tokens_per_s"]
    drift = abs(wall - exp) / max(wall, exp, 1) * 100
    res = {"ttft_p50": d["ttft_ms"]["p50"], "tpot_p50": d["tpot_ms"]["p50"],
           "tok_s": d["output_tokens_per_s"], "ok": d["successful_requests"],
           "total": d["planned_requests"], "wall_s": round(wall, 1),
           "expected_wall_s": round(exp, 1), "drift_pct": round(drift, 2),
           "self_consistent": drift <= 15.0}
    flag = "OK" if drift <= 15.0 else "DRIFT"
    log(f"{label}: TTFT={res['ttft_p50']:.0f}ms TPOT={res['tpot_p50']:.1f} "
        f"tok/s={res['tok_s']:.1f} wall={wall:.0f}s drift={drift:.1f}% {flag}")
    return res


def main():
    log("RESUME 5.3 (skip existing consistent points)")
    switches = [
        ("cpu_binding_off",
         ["--cpu-binding", "false", "--dsa-cp", "true",
          "--multistream-overlap", "true", "--static-kernel", "false"]),
        ("dsa_cp_off",
         ["--cpu-binding", "true", "--dsa-cp", "false",
          "--multistream-overlap", "true", "--static-kernel", "false"]),
        ("multistream_overlap_off",
         ["--cpu-binding", "true", "--dsa-cp", "true",
          "--multistream-overlap", "false", "--static-kernel", "false"]),
        ("static_kernel_on",
         ["--cpu-binding", "true", "--dsa-cp", "true",
          "--multistream-overlap", "true", "--static-kernel", "true"]),
    ]
    all_r = {}
    for name, sw in switches:
        out_dir = RD / f"add_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        all_r[name] = {}
        pending = []
        for case_id, inp, out, conc, nreq in CASES:
            label = f"add_{name}_{case_id}"
            oj = out_dir / f"{label}.json"
            if point_ok(str(oj)):
                d = json.load(open(oj))
                log(f"SKIP {label} (exists, consistent)")
                all_r[name][case_id] = {
                    "ttft_p50": d["ttft_ms"]["p50"],
                    "tpot_p50": d["tpot_ms"]["p50"],
                    "tok_s": d["output_tokens_per_s"],
                    "ok": d["successful_requests"], "total": d["planned_requests"],
                    "wall_s": d["wall_time_s"],
                    "expected_wall_s": round(d["total_output_tokens"] / d["output_tokens_per_s"], 1),
                    "drift_pct": 0.0, "self_consistent": True}
            else:
                pending.append((label, case_id, inp, out, conc, nreq))
        if not pending:
            log(f"{name}: all points present, skip server")
            continue
        log(f"{name}: {len(pending)} points pending -> restart server")
        pkill()
        env = os.environ.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
        srv_args = ["--mtp", "off", "--graph-mode", "cudagraph"] + sw
        logf = open(str(RD / f"add_{name}_server.log"), "w")
        subprocess.Popen([PYTHON, SERVER, "--port", "8000"] + srv_args,
                         env=env, stdout=logf, stderr=subprocess.STDOUT)
        if not wait_health(8000):
            all_r[name] = {"error": "server not healthy"}
            continue
        for label, case_id, inp, out, conc, nreq in pending:
            all_r[name][case_id] = run_point(label, out_dir, inp, out, conc, nreq)
        pkill()

    with open(str(RD / "5_3_add.json"), "w") as f:
        json.dump({"name": "5_3_add",
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "cases": all_r}, f, indent=2)
    log("RESUME 5.3 COMPLETE")


if __name__ == "__main__":
    main()

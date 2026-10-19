#!/usr/bin/env python3
"""Clean-session re-measurement of contaminated/unreliable latency data (Phase 2/3/4).

All benches run SEQUENTIALLY in one session, one config at a time. Every bench
is checked for wall-vs-throughput self-consistency (drift >15% => flagged).

Session blocks:
  A  MTP=off (production config): c8(32), c32(32), c32(128), c48(128), c64(128)
  B  MTP=1 : c8(32), c32(32), c32(128) + acceptance probe via /metrics
  C  two servers TP8x2 (devices 0-7 :8000, 8-15 :8001): c32(64 each), c64(64 each)

Results -> perf_opt/results/clean_final/
Run on remote:
    nohup python3 perf_opt/scripts/run_clean_final.py > perf_opt/results/clean_final/_session.log 2>&1 &
"""
import os, sys, subprocess, json, time

WORKDIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test"
os.chdir(WORKDIR)

BENCH = "perf_opt/scripts/bench_serving.py"
SERVER = "perf_opt/scripts/start_server_mtp.py"
# Results go to container-local /tmp: shared-disk project quota was full (2026-10-19).
RD = "/tmp/clean_final"
os.makedirs(RD, exist_ok=True)

session = {}


def log(msg):
    print(f"[clean] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def pkill():
    subprocess.run(["pkill", "-9", "-f", "VLLM::"], capture_output=True)
    time.sleep(5)


def start_server(mtp, port=8000, devices="0,1,2,3,4,5,6,7", logf=None):
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = devices
    cmd = [sys.executable, SERVER, "--port", str(port), "--mtp", mtp]
    with open(logf, "w") as l:
        subprocess.Popen(cmd, env=env, stdout=l, stderr=subprocess.STDOUT)
    for i in range(180):
        time.sleep(2)
        try:
            r = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                 f"http://127.0.0.1:{port}/health"],
                capture_output=True, text=True, timeout=3)
            if r.stdout.strip() == "200":
                log(f"server mtp={mtp} port={port} ready ({i*2}s)")
                return True
        except Exception:
            pass
    log(f"server mtp={mtp} port={port} FAILED")
    return False


def bench(label, c, nreq, port=8000):
    cid = f"{label}_c{c}_{nreq}req"
    oj = f"{RD}/{cid}.json"
    oc = f"{RD}/{cid}_npu.csv"
    cmd = [sys.executable, BENCH,
        "--base-url", f"http://127.0.0.1:{port}/v1",
        "--served-name", "dsv4",
        "--model-dir", "models/DeepSeek-V4-Flash-w8a8-self",
        "--input-len", "4096", "--output-len", "1024",
        "--concurrency", str(c), "--num-requests", str(nreq),
        "--case-id", cid, "--out", oj, "--npu-stats", oc,
        "--seed", "42", "--temp", "0.0"]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    dt = time.time() - t0
    if r.returncode != 0:
        log(f"{cid} FAIL rc={r.returncode} {r.stderr[-200:]}")
        return None
    d = json.load(open(oj))
    exp = d["total_output_tokens"] / d["output_tokens_per_s"]
    drift = (d["wall_time_s"] - exp) / exp
    flag = "WARN_DRIFT" if abs(drift) > 0.15 else "ok"
    log(f"{cid} done {dt:.0f}s wall={d['wall_time_s']:.0f}s exp={exp:.0f}s "
        f"drift={drift*100:+.1f}% ttft_p50={d['ttft_ms']['p50']} "
        f"tpot_p50={d['tpot_ms']['p50']} tok_s={d['output_tokens_per_s']} [{flag}]")
    return {
        "ttft_p50": d["ttft_ms"]["p50"], "ttft_p90": d["ttft_ms"]["p90"],
        "ttft_mean": d["ttft_ms"]["mean"], "tpot_p50": d["tpot_ms"]["p50"],
        "tok_s": d["output_tokens_per_s"], "wall_s": d["wall_time_s"],
        "expected_wall_s": round(exp, 1), "drift_pct": round(drift * 100, 1),
        "succ": d["successful_requests"], "total": d["planned_requests"],
        "self_consistent": abs(drift) <= 0.15,
    }


def probe_metrics(port=8000):
    try:
        r = subprocess.run(["curl", "-s", f"http://127.0.0.1:{port}/metrics"],
                           capture_output=True, text=True, timeout=5)
        draft = acc = 0
        for ln in r.stdout.split("\n"):
            if "spec_decode_num_draft_tokens_total" in ln and not ln.startswith("#"):
                draft = float(ln.split()[-1])
            if "spec_decode_num_accepted_tokens_total" in ln and not ln.startswith("#"):
                acc = float(ln.split()[-1])
        return {"draft": draft, "accepted": acc}
    except Exception as e:
        return {"error": str(e)}


def main():
    session["started"] = time.strftime("%F %T")
    session["node"] = os.popen("hostname").read().strip()

    # ---- Block A: production config (MTP off) ----
    pkill()
    if start_server("off", logf=f"{RD}/_srv_off.log"):
        session["A_mtp_off"] = {}
        for c, n in [(8, 32), (32, 32), (32, 128), (48, 128), (64, 128)]:
            session["A_mtp_off"][f"c{c}_{n}req"] = bench("off", c, n)
    pkill()

    # ---- Block B: MTP=1 ----
    if start_server("1", logf=f"{RD}/_srv_on.log"):
        pre = probe_metrics()
        session["B_mtp1"] = {}
        for c, n in [(8, 32), (32, 32), (32, 128)]:
            session["B_mtp1"][f"c{c}_{n}req"] = bench("mtp1", c, n)
        post = probe_metrics()
        dd = post.get("draft", 0) - pre.get("draft", 0)
        da = post.get("accepted", 0) - pre.get("accepted", 0)
        session["B_mtp1"]["acceptance"] = (
            round(da / dd, 4) if dd > 0 else None)
        log(f"acceptance: {da:.0f}/{dd:.0f} = {session['B_mtp1']['acceptance']}")
    pkill()

    # ---- Block C: two servers TP8x2 ----
    if start_server("off", port=8000, devices="0,1,2,3,4,5,6,7", logf=f"{RD}/_srv_dp0.log") and \
       start_server("off", port=8001, devices="8,9,10,11,12,13,14,15", logf=f"{RD}/_srv_dp1.log"):
        session["C_tp8x2"] = {}
        for c, n in [(32, 64), (64, 64)]:
            session["C_tp8x2"][f"dp0_c{c}_{n}req"] = bench("dp0", c, n, port=8000)
            session["C_tp8x2"][f"dp1_c{c}_{n}req"] = bench("dp1", c, n, port=8001)
            # aggregate
            try:
                a = session["C_tp8x2"][f"dp0_c{c}_{n}req"]
                b = session["C_tp8x2"][f"dp1_c{c}_{n}req"]
                if a and b:
                    session["C_tp8x2"][f"agg_c{c}"] = {
                        "tok_s_sum": round(a["tok_s"] + b["tok_s"], 1),
                        "ttft_p50_balance": [a["ttft_p50"], b["ttft_p50"]],
                    }
            except Exception:
                pass
    pkill()

    session["finished"] = time.strftime("%F %T")
    with open(f"{RD}/_session_summary.json", "w") as f:
        json.dump(session, f, indent=2)
    log("ALL DONE")


if __name__ == "__main__":
    main()

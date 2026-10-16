#!/usr/bin/env python3
"""Phase 4 full MTP matrix benchmark — multi-config (off/1/2) × multi-concurrency (1/8/32)."""

import os, sys, subprocess, json, time

WORKDIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test"
os.chdir(WORKDIR)

BENCH = "perf_opt/scripts/bench_serving.py"
SERVER = "perf_opt/scripts/start_server_mtp.py"
RESULTS_DIR = "perf_opt/results/mtp"
PORT = 8000

MATRIX = {
    "off": [8, 32],    # c=1 done
    "1":   [8, 32],    # c=1 done
    "2":   [1, 8, 32], # all
}


def run_bench(concurrency, case_id, out_json, out_csv):
    t0 = time.time()
    cmd = [sys.executable, BENCH,
        "--base-url", f"http://127.0.0.1:{PORT}/v1",
        "--served-name", "dsv4",
        "--model-dir", "models/DeepSeek-V4-Flash-w8a8-self",
        "--input-len", "4096", "--output-len", "1024",
        "--concurrency", str(concurrency), "--num-requests", "32",
        "--case-id", case_id, "--out", out_json, "--npu-stats", out_csv,
        "--seed", "42", "--temp", "0.0"]
    print(f"[matrix] {case_id} START", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    dt = time.time() - t0
    if r.returncode == 0:
        with open(out_json) as f:
            d = json.load(f)
        print(f"[matrix] {case_id} DONE {dt:.0f}s "
              f"succ={d['successful_requests']}/{d['planned_requests']}", flush=True)
        return True
    print(f"[matrix] {case_id} FAIL rc={r.returncode}", flush=True)
    return False


def start_server(mtp):
    subprocess.run(["pkill", "-9", "-f", "VLLM::"], capture_output=True)
    time.sleep(5)
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    logf = f"{RESULTS_DIR}/_srv_mtp{mtp}.log"
    cmd = [sys.executable, SERVER, "--port", str(PORT), "--mtp", mtp]
    with open(logf, "w") as l:
        subprocess.Popen(cmd, env=env, stdout=l, stderr=subprocess.STDOUT)
    for i in range(180):
        time.sleep(2)
        try:
            r = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                 f"http://127.0.0.1:{PORT}/health"],
                capture_output=True, text=True, timeout=3)
            if r.stdout.strip() == "200":
                print(f"[matrix] server MTP={mtp} ready ({i*2}s)", flush=True)
                return True
        except Exception:
            pass
    return False


def probe_accept():
    try:
        r = subprocess.run(["curl", "-s", f"http://127.0.0.1:{PORT}/metrics"],
                          capture_output=True, text=True, timeout=5)
        draft = acc = 0
        for ln in r.stdout.split("\n"):
            if "spec_decode_num_draft_tokens_total" in ln and not ln.startswith("#"):
                draft = float(ln.split()[-1])
            if "spec_decode_num_accepted_tokens_total" in ln and not ln.startswith("#"):
                acc = float(ln.split()[-1])
        if draft > 0:
            return {"draft": draft, "accepted": acc, "rate": acc / draft}
    except Exception:
        pass
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {}
    for mtp, cs in MATRIX.items():
        d = f"{RESULTS_DIR}/mtp_{mtp}"
        os.makedirs(d, exist_ok=True)
        if not start_server(mtp):
            summary[mtp] = {"error": "server_failed"}
            continue
        acc_pre = probe_accept()
        summary[mtp] = {"acceptance": acc_pre["rate"] if acc_pre and mtp != "off" else None}
        for c in cs:
            cid = f"mtp{mtp}_c{c}"
            oj = f"{d}/mtp_{mtp}_probe_c{c}.json"
            oc = f"{d}/mtp_{mtp}_probe_c{c}_npu.csv"
            if os.path.exists(oj):
                with open(oj) as f:
                    ex = json.load(f)
                if ex.get("successful_requests") == ex.get("planned_requests"):
                    print(f"[matrix] SKIP {cid}", flush=True)
                    summary[mtp][f"c{c}"] = {
                        "ttft_p50": ex["ttft_ms"]["p50"],
                        "tpot_p50": ex["tpot_ms"]["p50"],
                        "tok_s": ex["output_tokens_per_s"],
                        "wall_s": ex["wall_time_s"],
                    }
                    continue
            ok = run_bench(c, cid, oj, oc)
            if ok:
                with open(oj) as f:
                    ex = json.load(f)
                summary[mtp][f"c{c}"] = {
                    "ttft_p50": ex["ttft_ms"]["p50"],
                    "tpot_p50": ex["tpot_ms"]["p50"],
                    "tok_s": ex["output_tokens_per_s"],
                    "wall_s": ex["wall_time_s"],
                }
            else:
                summary[mtp][f"c{c}"] = {"error": "bench_failed"}
        acc_post = probe_accept()
        if acc_post and mtp != "off" and acc_pre:
            dd = acc_post["draft"] - acc_pre["draft"]
            da = acc_post["accepted"] - acc_pre["accepted"]
            if dd > 0:
                summary[mtp]["acceptance"] = da / dd
                print(f"[matrix] MTP={mtp} acceptance={da/dd:.4f}", flush=True)
    sp = "perf_opt/summary/mtp_best.json"
    os.makedirs(os.path.dirname(sp), exist_ok=True)
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[matrix] ALL DONE -> {sp}", flush=True)
    subprocess.run(["pkill", "-9", "-f", "VLLM::"], capture_output=True)


if __name__ == "__main__":
    main()
    return None
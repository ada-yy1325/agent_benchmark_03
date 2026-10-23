#!/usr/bin/env python3
"""mtp_model_suite.py — 好模型(w8a8-mtp)完整复测 + 归档编排.

Blocks (sequential, every bench wall/throughput self-checked):
 1. consistency: determinism check (2 prompts x 3 runs, temp=0 seed=42)
 2. gpqa smoke (3 questions) on final config
 3. gpqa full (198) on final config
 4. production config (static_kernel ON, kv fp8) 4 frozen points
 5. baseline (static_kernel OFF) 2 points (c1, c32)
 6. MTP on: c1/c8/c32 + acceptance via /metrics deltas
 7. two-server TP8x2: c32 (64req each)
 8. kickoff: npu16 re-quant (16 NPUs) in background, then exit

Results -> /tmp/phase5/mtp_model/ (fetched to repo after completion)
Run: setsid nohup python3 /tmp/mtp_model_suite.py > /tmp/phase5/mtp_model/_suite.log 2>&1 < /dev/null &
"""
import os, sys, subprocess, json, time
from pathlib import Path

WORKDIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test"
os.chdir(WORKDIR)

BENCH = str(Path("perf_opt/scripts/bench_serving.py").resolve())
SERVER = "/tmp/start_server_mtp.py"  # phase5 launcher with MODEL_DIR -> w8a8-mtp
PYTHON = "/opt/mamba/bin/python3"
MODEL_DIR = "/inspire/sj-ssd3/global_public/public_models/DeepSeek-V4-Flash-w8a8-mtp"
RD = Path("/tmp/phase5/mtp_model")
RD.mkdir(parents=True, exist_ok=True)

CASES = [
    ("c1_4k", 4096, 1024, 1, 16),
    ("c8_4k", 4096, 1024, 8, 64),
    ("c32_4k", 4096, 1024, 32, 128),
    ("c64_4k", 4096, 1024, 64, 128),
]


def log(msg):
    print(f"[suite] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def pkill():
    subprocess.run(["pkill", "-9", "-f", "VLLM::"], capture_output=True)
    time.sleep(5)


def wait_health(port, timeout=400):
    for i in range(timeout // 2):
        time.sleep(2)
        try:
            r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                                f"http://127.0.0.1:{port}/health"],
                               capture_output=True, text=True, timeout=10)
            if r.stdout.strip() == "200":
                log(f"server healthy port={port} ({(i + 1) * 2}s)")
                return True
        except Exception:
            pass
    log(f"server port={port} NOT healthy")
    return False


def start_server(port, extra_args, devices, logf):
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = devices
    cmd = [PYTHON, SERVER, "--port", str(port)] + extra_args
    f = open(logf, "w")
    subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    log(f"launched server port={port} args={' '.join(extra_args)}")
    return wait_health(port)


def run_bench(label, port, inp, out, conc, nreq):
    out_json = str(RD / f"{label}.json")
    npu_csv = str(RD / f"{label}_npu.csv")
    bench_log = str(RD / f"{label}_bench.log")
    cmd = [PYTHON, BENCH, "--served-name", "dsv4", "--model-dir", MODEL_DIR,
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
        log(f"FAIL {label} rc={r.returncode}")
        return {"error": f"rc={r.returncode}", "wall_s": round(wall, 1)}
    d = json.load(open(out_json))
    exp = d["total_output_tokens"] / d["output_tokens_per_s"]
    drift = abs(wall - exp) / max(wall, exp, 1) * 100
    res = {"ttft_p50": d["ttft_ms"]["p50"], "tpot_p50": d["tpot_ms"]["p50"],
           "tok_s": d["output_tokens_per_s"], "ok": d["successful_requests"],
           "total": d["planned_requests"], "wall_s": round(wall, 1),
           "expected_wall_s": round(exp, 1), "drift_pct": round(drift, 2),
           "self_consistent": drift <= 15.0}
    log(f"{label}: TTFT={res['ttft_p50']:.0f} TPOT={res['tpot_p50']:.1f} "
        f"tok/s={res['tok_s']:.1f} wall={wall:.0f}s drift={drift:.1f}% "
        f"{'OK' if drift <= 15 else 'DRIFT'}")
    return res


def probe_metrics(port):
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


def chat(port, prompt, max_tok=64, seed=42):
    import urllib.request
    payload = json.dumps({"model": "dsv4",
                          "messages": [{"role": "user", "content": prompt}],
                          "temperature": 0.0, "max_tokens": max_tok, "seed": seed}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"]


def block_consistency(port):
    log("=== BLOCK 1: consistency ===")
    prompts = ["What is the capital of France? Answer in one word.",
               "If 5x + 3 = 28, what is x? Show your steps briefly."]
    out = {}
    for i, p in enumerate(prompts):
        runs = [chat(port, p, 64) for _ in range(3)]
        same = all(x == runs[0] for x in runs)
        out[f"p{i}"] = {"same": same, "runs": runs}
        log(f"p{i} deterministic={same} sample={runs[0][:60]!r}")
    with open(str(RD / "consistency.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def block_gpqa(smoke):
    log(f"=== BLOCK 2/3: gpqa {'smoke' if smoke else 'FULL'} ===")
    script = f"""
import json, os, sys, time
API_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "dsv4"
SMOKE = {smoke}
from evalscope import TaskConfig, run_task
gen_cfg = {{'temperature': 0.0, 'max_tokens': 8192, 'n': 1, 'seed': 42}}
task_kwargs = dict(model=MODEL, api_url=API_URL, eval_type='openai_api',
    datasets=['gpqa_diamond'], dataset_args={{'gpqa_diamond': {{}}}},
    eval_batch_size=8, generation_config=gen_cfg, timeout=300000, stream=True)
if SMOKE:
    task_kwargs['limit'] = SMOKE
t0 = time.time()
run_task(task_cfg=TaskConfig(**task_kwargs))
dt = time.time() - t0
outputs_dir = "./outputs"
dirs = sorted([d for d in os.listdir(outputs_dir)
               if os.path.isdir(os.path.join(outputs_dir, d)) and d.startswith("2026")])
latest = os.path.join(outputs_dir, dirs[-1])
reviews_dir = os.path.join(latest, "reviews", MODEL)
files = [f for f in os.listdir(reviews_dir) if f.endswith(".jsonl")] if os.path.isdir(reviews_dir) else []
if files:
    jpath = os.path.join(reviews_dir, files[0])
    correct = total = 0
    for line in open(jpath):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        total += 1
        sc = d.get("sample_score", {{}})
        acc = sc.get("score", {{}}).get("value", {{}}).get("acc", 0.0)
        if acc and acc > 0:
            correct += 1
    res = {{"model": MODEL, "dataset": "gpqa_diamond", "num_questions": total,
            "correct": correct, "accuracy_pct": round(correct/total*100, 2),
            "elapsed_s": round(dt, 1), "eval_dir": latest}}
    with open("{RD}/gpqa_{'smoke' if SMOKE else 'full'}_result.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"[gpqa] ACCURACY: {{correct}}/{{total}} = {{correct/total*100:.2f}}%", flush=True)
else:
    print("[gpqa] no reviews found; latest:", latest, flush=True)
"""
    with open(str(RD / f"gpqa_{'smoke' if smoke else 'full'}.log"), "w") as lf:
        r = subprocess.run([PYTHON, "-c", script], stdout=lf, stderr=subprocess.STDOUT, cwd=WORKDIR, timeout=14400)
    log(f"gpqa {'smoke' if smoke else 'full'} done rc={r.returncode}")
    return r.returncode == 0


def main():
    summary = {"model": MODEL_DIR, "started": time.strftime("%F %T")}

    # ---- final config server (static_kernel ON, kv fp8) ----
    pkill()
    if start_server(8000, ["--mtp", "off", "--static-kernel", "true", "--kv-cache-dtype", "fp8"],
                    "0,1,2,3,4,5,6,7", str(RD / "_srv_final.log")):
        block_consistency(8000)
        block_gpqa(smoke=3)
        block_gpqa(smoke=0)
    pkill()

    # ---- production config 4 points ----
    log("=== BLOCK 4: production config ===")
    if start_server(8000, ["--mtp", "off", "--static-kernel", "true", "--kv-cache-dtype", "fp8"],
                    "0,1,2,3,4,5,6,7", str(RD / "_srv_prod.log")):
        summary["prod"] = {}
        for case_id, inp, out, conc, nreq in CASES:
            summary["prod"][case_id] = run_bench(f"prod_{case_id}", 8000, inp, out, conc, nreq)
    pkill()

    # ---- baseline 2 points (static_kernel OFF) ----
    log("=== BLOCK 5: baseline (static_kernel off) ===")
    if start_server(8000, ["--mtp", "off"], "0,1,2,3,4,5,6,7", str(RD / "_srv_base.log")):
        summary["base"] = {}
        for case_id, inp, out, conc, nreq in CASES:
            if case_id not in ("c1_4k", "c32_4k"):
                continue
            summary["base"][case_id] = run_bench(f"base_{case_id}", 8000, inp, out, conc, nreq)
    pkill()

    # ---- MTP on: 3 points + acceptance ----
    log("=== BLOCK 6: MTP on ===")
    if start_server(8000, ["--mtp", "1"], "0,1,2,3,4,5,6,7", str(RD / "_srv_mtpon.log")):
        pre = probe_metrics(8000)
        summary["mtpon"] = {}
        for case_id, inp, out, conc, nreq in CASES:
            if case_id == "c64_4k":
                continue
            summary["mtpon"][case_id] = run_bench(f"mtpon_{case_id}", 8000, inp, out, conc, nreq)
        post = probe_metrics(8000)
        dd = post.get("draft", 0) - pre.get("draft", 0)
        da = post.get("accepted", 0) - pre.get("accepted", 0)
        summary["mtpon"]["acceptance"] = round(da / dd, 4) if dd > 0 else None
        log(f"acceptance: {da:.0f}/{dd:.0f} = {summary['mtpon']['acceptance']}")
    pkill()

    # ---- two-server c32 ----
    log("=== BLOCK 7: two-server TP8x2 c32 ===")
    ok0 = start_server(8000, ["--mtp", "off", "--static-kernel", "true", "--kv-cache-dtype", "fp8"],
                       "0,1,2,3,4,5,6,7", str(RD / "_srv_dp0.log"))
    ok1 = start_server(8001, ["--mtp", "off", "--static-kernel", "true", "--kv-cache-dtype", "fp8"],
                       "8,9,10,11,12,13,14,15", str(RD / "_srv_dp1.log"))
    if ok0 and ok1:
        summary["tp8x2"] = {
            "dp0": run_bench("tp8x2_dp0_c32", 8000, 4096, 1024, 32, 64),
            "dp1": run_bench("tp8x2_dp1_c32", 8001, 4096, 1024, 32, 64),
        }
    pkill()

    summary["finished"] = time.strftime("%F %T")
    with open(str(RD / "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- BLOCK 8: kickoff npu16 re-quant ----
    log("=== BLOCK 8: kickoff npu16 re-quant (16 NPUs, ~3h) ===")
    quant_script = "archive/dsv4_quant/quant_dsv4_npu16.py"
    qlog = open("/tmp/phase5/requant_npu16.log", "w")
    subprocess.Popen(["setsid", PYTHON, quant_script],
                     stdout=qlog, stderr=subprocess.STDOUT,
                     start_new_session=True)
    log("quant launched -> /tmp/phase5/requant_npu16.log")
    log("ALL DONE")


if __name__ == "__main__":
    main()

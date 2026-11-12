#!/usr/bin/env python3
"""
SLO 约束并发爬坡基准套件 —— 判定口径 p90
TTFT p90 < 5000 ms / TPOT p90 < 30 ms
任一越线 = 该并发点"劣化"；最大未越线并发 = SLO 可用并发，该点吞吐 = goodput

Usage (on remote):
    # PD 部署 (P/D 已由 launch.sh 启动)
    python3 perf_opt/scripts/slo_benchmark.py \\
        --deployment pd_mtp --scenario 16k1k \\
        --base-url http://127.0.0.1:8000/v1 \\
        --model-dir /inspire/.../models/DeepSeek-V4-Flash-w8a8-npu16

    # 单服 TP8+MTP
    python3 perf_opt/scripts/slo_benchmark.py \\
        --deployment single_mtp --scenario 16k1k \\
        --base-url http://127.0.0.1:8000/v1 --model-dir ...

    # 双 TP8+MTP
    python3 perf_opt/scripts/slo_benchmark.py \\
        --deployment dual_mtp --scenario 16k1k \\
        --base-url http://127.0.0.1:8000/v1 \\
        --base-url-b http://127.0.0.1:8001/v1 --model-dir ...
"""

import argparse, json, os, subprocess, sys, time
from datetime import datetime

# ── 常量 ──
RESULTS_ROOT = "pd_deployment/results/slo_scan"
PROGRESS_FILE = "pd_deployment/results/slo_scan/progress.json"
BENCH_SCRIPT = "perf_opt/scripts/bench_serving.py"
SERVED_NAME = "dsv4"
SCENARIOS = {
    "16k1k": {"input_len": 16384, "output_len": 1024, "label": "16K→1K"},
    "4k1k":  {"input_len": 4096,  "output_len": 1024, "label": "4K→1K"},
}
CONCURRENCIES = [8, 16, 32, 48, 64, 96, 128]
SLO_TTFT_P90 = 5_000   # ms
SLO_TPOT_P90 = 30      # ms


def req_count(c: int) -> int:
    """请求数：c≤16 用 16，c>16 用 2×c"""
    return max(16, 2 * c)


def run_single(args_dict: dict) -> dict:
    """Run bench_serving.py once, return parsed result dict."""
    cmd = [
        sys.executable, BENCH_SCRIPT,
        "--base-url", args_dict["base_url"],
        "--served-name", SERVED_NAME,
        "--model-dir", args_dict["model_dir"],
        "--case-id", args_dict["case_id"],
        "--out", args_dict["out"],
        "--npu-stats", args_dict["npu_stats"],
        "--input-len", str(args_dict["input_len"]),
        "--output-len", str(args_dict["output_len"]),
        "--concurrency", str(args_dict["concurrency"]),
        "--num-requests", str(args_dict["num_requests"]),
        "--temp", "0",
        "--seed", "42",
    ]
    print(f"[slo] RUN: {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, capture_output=False, timeout=3600)
    if p.returncode != 0:
        print(f"[slo] FAIL: bench_serving.py returned {p.returncode}", flush=True)
        return None
    try:
        with open(args_dict["out"]) as f:
            return json.load(f)
    except Exception as e:
        print(f"[slo] FAIL reading {args_dict['out']}: {e}", flush=True)
        return None
def combine_dual_results(r_a: dict, r_b: dict) -> dict:
    """Merge two bench results (dual TP8) into one aggregated result."""
    na, nb = r_a["planned_requests"], r_b["planned_requests"]
    sa, sb = r_a["successful_requests"], r_b["successful_requests"]
    total_ok = sa + sb
    total_out = r_a["total_output_tokens"] + r_b["total_output_tokens"]
    total_in = r_a["total_input_tokens"] + r_b["total_input_tokens"]
    wall = max(r_a["wall_time_s"], r_b["wall_time_s"])
    throughput = round(total_out / wall, 2) if wall > 0 else 0

    if total_ok == 0:
        return r_a
    wa, wb = sa / total_ok, sb / total_ok

    def w_pct(key, p):
        return round(wa * r_a[key]["p" + str(p)] + wb * r_b[key]["p" + str(p)], 2)

    def w_mean(key):
        return round(wa * r_a[key]["mean"] + wb * r_b[key]["mean"], 2)

    return {
        "case_id": r_a["case_id"],
        "input_len": r_a["input_len"], "output_len": r_a["output_len"],
        "concurrency": r_a["concurrency"] + r_b["concurrency"],
        "planned_requests": na + nb,
        "successful_requests": total_ok,
        "failed_requests": (na - sa) + (nb - sb),
        "wall_time_s": wall,
        "total_input_tokens": total_in, "total_output_tokens": total_out,
        "output_tokens_per_s": throughput,
        "ttft_ms": {
            "p50": w_pct("ttft_ms", 50), "p90": w_pct("ttft_ms", 90),
            "p99": w_pct("ttft_ms", 99), "mean": w_mean("ttft_ms"),
        },
        "tpot_ms": {
            "p50": w_pct("tpot_ms", 50), "p90": w_pct("tpot_ms", 90),
            "p99": w_pct("tpot_ms", 99), "mean": w_mean("tpot_ms"),
        },
        "success_rate": round(total_ok / (na + nb), 4),
        "aggregated_from_dual": True,
        "timestamp": datetime.now().isoformat(),
    }


def check_slo(r: dict) -> dict:
    ttft_ok = r["ttft_ms"]["p90"] < SLO_TTFT_P90
    tpot_ok = r["tpot_ms"]["p90"] < SLO_TPOT_P90
    return {"ttft": ttft_ok, "tpot": tpot_ok, "pass": ttft_ok and tpot_ok}


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}


def save_progress(progress):
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)
def main():
    ap = argparse.ArgumentParser(description="SLO-约束并发爬坡")
    ap.add_argument("--deployment", required=True,
                    choices=["pd_nomtp", "pd_mtp", "single_mtp", "dual_mtp"])
    ap.add_argument("--scenario", required=True, choices=["16k1k", "4k1k"])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--base-url-b", default="http://127.0.0.1:8001/v1",
                    help="Second server for dual TP8")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    scenario = SCENARIOS[args.scenario]
    out_dir = os.path.join(RESULTS_ROOT, args.deployment, args.scenario)
    os.makedirs(out_dir, exist_ok=True)

    progress = load_progress()
    key = f"{args.deployment}/{args.scenario}"
    done = set(progress.get(key, []))

    if args.dry_run:
        print(f"[DRY RUN] {key}", flush=True)
        for c in CONCURRENCIES:
            cid = f"{args.deployment}_{args.scenario}_c{c}"
            print(f"  c={c:4d} reqs={req_count(c):4d} {'SKIP' if cid in done else 'RUN'}", flush=True)
        return 0

    print(f"[slo] === {args.deployment} / {scenario['label']} ===", flush=True)
    print(f"[slo] SLO: TTFT p90 < {SLO_TTFT_P90}ms, TPOT p90 < {SLO_TPOT_P90}ms", flush=True)

    con_fail = 0

    for c in CONCURRENCIES:
        cid = f"{args.deployment}_{args.scenario}_c{c}"
        if cid in done:
            print(f"[slo] c={c:4d} ⏭ skip", flush=True)
            continue

        n_req = req_count(c)
        t0 = time.time()
        print(f"[slo] c={c:4d} reqs={n_req} ▶ running...", flush=True)

        if args.deployment == "dual_mtp":
            c_per = c // 2
            n_per = req_count(c_per)
            a_out = os.path.join(out_dir, f"{cid}_a.json")
            b_out = os.path.join(out_dir, f"{cid}_b.json")
            a_npu = os.path.join(out_dir, f"{cid}_a_npu.csv")
            b_npu = os.path.join(out_dir, f"{cid}_b_npu.csv")
            args_a = dict(base_url=args.base_url, model_dir=args.model_dir,
                          case_id=f"{cid}_a", out=a_out, npu_stats=a_npu,
                          input_len=scenario["input_len"],
                          output_len=scenario["output_len"],
                          concurrency=c_per, num_requests=n_per)
            args_b = dict(base_url=args.base_url_b, model_dir=args.model_dir,
                          case_id=f"{cid}_b", out=b_out, npu_stats=b_npu,
                          input_len=scenario["input_len"],
                          output_len=scenario["output_len"],
                          concurrency=c_per, num_requests=n_per)
            r_a = run_single(args_a)
            r_b = run_single(args_b)
            if r_a is None or r_b is None:
                print(f"[slo] c={c:4d} ✗ FAIL", flush=True)
                break
            result = combine_dual_results(r_a, r_b)
            combined = os.path.join(out_dir, f"{cid}_combined.json")
            with open(combined, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        else:
            f_out = os.path.join(out_dir, f"{cid}.json")
            f_npu = os.path.join(out_dir, f"{cid}_npu.csv")
            s_args = dict(base_url=args.base_url, model_dir=args.model_dir,
                          case_id=cid, out=f_out, npu_stats=f_npu,
                          input_len=scenario["input_len"],
                          output_len=scenario["output_len"],
                          concurrency=c, num_requests=n_req)
            result = run_single(s_args)
            if result is None:
                print(f"[slo] c={c:4d} ✗ FAIL", flush=True)
                break

        elapsed = time.time() - t0
        slo = check_slo(result)
        ttft_p90 = result["ttft_ms"]["p90"]
        tpot_p90 = result["tpot_ms"]["p90"]
        tp = result["output_tokens_per_s"]
        succ = result["successful_requests"]
        total = result["planned_requests"]

        print(f"[slo] c={c:4d} ✓ {succ}/{total} "
              f"TTFT p50/p90/p99={result['ttft_ms']['p50']}/{ttft_p90}/{result['ttft_ms']['p99']}ms "
              f"TPOT p50/p90/p99={result['tpot_ms']['p50']}/{tpot_p90}/{result['tpot_ms']['p99']}ms "
              f"tok/s={tp} wall={elapsed:.0f}s", flush=True)

        flags = []
        if not slo["ttft"]:
            flags.append(f"TTFT p90={ttft_p90}≥{SLO_TTFT_P90}")
        if not slo["tpot"]:
            flags.append(f"TPOT p90={tpot_p90}≥{SLO_TPOT_P90}")
        if flags:
            print(f"[slo] c={c:4d} ⛔ {' / '.join(flags)}", flush=True)

        done.add(cid)
        progress[key] = list(done)
        save_progress(progress)

        if not slo["pass"] and not slo["ttft"] and not slo["tpot"]:
            con_fail += 1
        else:
            con_fail = 0

        if con_fail >= 2:
            print(f"[slo] STOP: 2 consecutive full SLO failures", flush=True)
            break

    print(f"[slo] DONE {args.deployment}/{args.scenario}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
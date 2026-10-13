#!/usr/bin/env python3
"""
Phase 1.5: 矩阵汇总

遍历 results/matrix/*.json，生成：
  - summary/matrix_summary.json（完整元数据）
  - summary/matrix_summary.csv（一页纵览）
"""

import json, os, glob, csv
from datetime import datetime

BASE_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/perf_opt"
RESULTS_DIR = os.path.join(BASE_DIR, "results", "matrix")
SUMMARY_DIR = os.path.join(BASE_DIR, "summary")


def load_cases():
    cases = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        try:
            with open(path) as f:
                cases.append(json.load(f))
        except Exception as e:
            print(f"⚠ skip {path}: {e}")
    return cases


def make_summary_json(cases):
    by_input = {}
    for c in cases:
        il = c["input_len"]
        by_input.setdefault(il, []).append(c)

    aggregates = {}
    for il, case_list in sorted(by_input.items()):
        case_list.sort(key=lambda x: x["concurrency"])
        successful = [c for c in case_list if c["success_rate"] >= 0.95]

        peak_tok = max(c["output_tokens_per_s"] for c in case_list)
        peak_case = next(c for c in case_list if c["output_tokens_per_s"] == peak_tok)

        sweet_cases = sorted(
            [c for c in successful if c["output_tokens_per_s"] >= peak_tok * 0.95],
            key=lambda x: x["concurrency"], reverse=True
        )
        sweet = sweet_cases[0] if sweet_cases else None

        degrade = None
        for c in case_list:
            if c["success_rate"] < 0.95:
                degrade = c; break
            if peak_tok > 0 and c["output_tokens_per_s"] < peak_tok * 0.8:
                degrade = c; break

        tag = f"{il // 1024}K" if il % 1024 == 0 else str(il)
        aggregates[tag] = {
            "input_len": il, "num_cases": len(case_list),
            "successful_cases": len(successful),
            "peak_output_tok_s": peak_tok,
            "peak_concurrency": peak_case["concurrency"],
            "sweet_spot_concurrency": sweet["concurrency"] if sweet else None,
            "degradation_concurrency": degrade["concurrency"] if degrade else None,
            "degradation_reason": "success_rate" if degrade and degrade["success_rate"] < 0.95
            else ("throughput_drop" if degrade else None),
        }

    return {"generated": datetime.now().isoformat(), "num_cases": len(cases),
            "cases": cases, "aggregates": aggregates}


def make_csv(cases):
    fields = [
        "case_id", "input_len", "concurrency",
        "planned_requests", "successful_requests", "failed_requests",
        "success_rate", "wall_time_s",
        "output_tokens_per_s", "total_tokens_per_s",
        "ttft_p50_ms", "ttft_p99_ms",
        "tpot_p50_ms", "tpot_p99_ms",
        "e2e_p50_ms", "e2e_p99_ms",
        "peak_hbm_mb_per_npu",
    ]
    rows = []
    for c in cases:
        peak_hbm = c.get("peak_hbm_mb_per_npu", [])
        rows.append({
            "case_id": c.get("case_id", ""),
            "input_len": c.get("input_len", 0),
            "concurrency": c.get("concurrency", 0),
            "planned_requests": c.get("planned_requests", 0),
            "successful_requests": c.get("successful_requests", 0),
            "failed_requests": c.get("failed_requests", 0),
            "success_rate": round(c.get("success_rate", 0), 4),
            "wall_time_s": c.get("wall_time_s", 0),
            "output_tokens_per_s": c.get("output_tokens_per_s", 0),
            "total_tokens_per_s": c.get("total_tokens_per_s", 0),
            "ttft_p50_ms": c.get("ttft_ms", {}).get("p50", 0),
            "ttft_p99_ms": c.get("ttft_ms", {}).get("p99", 0),
            "tpot_p50_ms": c.get("tpot_ms", {}).get("p50", 0),
            "tpot_p99_ms": c.get("tpot_ms", {}).get("p99", 0),
            "e2e_p50_ms": c.get("e2e_ms", {}).get("p50", 0),
            "e2e_p99_ms": c.get("e2e_ms", {}).get("p99", 0),
            "peak_hbm_mb_per_npu": ",".join(map(str, peak_hbm)) if peak_hbm else "",
        })
    return fields, rows


def main():
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    print("Loading cases...")
    cases = load_cases()
    print(f"Found {len(cases)} cases")
    if not cases:
        print("No cases. Exit.")
        return
    cases.sort(key=lambda x: (x.get("input_len", 0), x.get("concurrency", 0)))

    summary = make_summary_json(cases)
    with open(os.path.join(SUMMARY_DIR, "matrix_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    fields, rows = make_csv(cases)
    with open(os.path.join(SUMMARY_DIR, "matrix_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\n── 快速纵览 ──")
    for row in rows:
        status = "✅" if row["successful_requests"] == row["planned_requests"] else "⚠"
        print(f"  {status} {row['case_id']:20s} c={row['concurrency']:2d} "
              f"out_tok/s={row['output_tokens_per_s']:7.2f} "
              f"TTFT_p50={row['ttft_p50_ms']:7.1f}ms "
              f"TPOT_p50={row['tpot_p50_ms']:7.1f}ms "
              f"rate={row['success_rate']:.2%}")
    print("\n── 各档汇总 ──")
    for tag, agg in summary.get("aggregates", {}).items():
        print(f"  {tag}: peak={agg['peak_output_tok_s']:.1f}/s @c={agg['peak_concurrency']}, "
              f"sweet_c={agg['sweet_spot_concurrency']}, "
              f"degrade_c={agg['degradation_concurrency']} ({agg['degradation_reason'] or '-'})")


if __name__ == "__main__":
    main()

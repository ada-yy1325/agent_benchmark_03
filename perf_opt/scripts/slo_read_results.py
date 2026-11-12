#!/usr/bin/env python3
"""
SLO 爬坡结果分析器 —— 读取 slo_scan/*.json，生成爬坡表和 SLO 结论表。
Usage: python3 perf_opt/scripts/slo_read_results.py
"""

import json, os, sys

RESULTS_ROOT = "pd_deployment/results/slo_scan"
SLO_TTFT = 5_000
SLO_TPOT = 30


def load_all():
    """Return {deployment: {scenario: [c_results]}}"""
    data = {}
    for dep in sorted(os.listdir(RESULTS_ROOT)):
        dep_path = os.path.join(RESULTS_ROOT, dep)
        if not os.path.isdir(dep_path):
            continue
        data[dep] = {}
        for sc in sorted(os.listdir(dep_path)):
            sc_path = os.path.join(dep_path, sc)
            if not os.path.isdir(sc_path):
                continue
            points = []
            for fname in sorted(os.listdir(sc_path)):
                if not fname.endswith(".json") or "npu" in fname:
                    continue
                if "_a.json" in fname or "_b.json" in fname:
                    continue
                with open(os.path.join(sc_path, fname)) as f:
                    points.append(json.load(f))
            points.sort(key=lambda x: x["concurrency"])
            data[dep][sc] = points
    return data


def climb_table(deployment, scenario, points):
    """Print a climb table. Returns {slo_concurrency, goodput}."""
    label = f"{deployment} / {scenario}"
    print(f"\n{'='*130}")
    print(f"  {label}")
    print(f"{'='*130}")
    hdr = f"{'c':>5} {'ok':>5} {'TTFTp50':>9} {'TTFTp90':>9} {'TTFTp99':>9} {'TPOTp50':>8} {'TPOTp90':>8} {'TPOTp99':>8} {'tok/s':>9} {'TTFT':>6} {'TPOT':>6}"
    print(hdr)
    print("-" * 130)

    good_c, good_tp = None, None
    for p in points:
        c = p["concurrency"]
        s, t = p["successful_requests"], p["planned_requests"]
        t90 = p["ttft_ms"]["p90"]
        tp90 = p["tpot_ms"]["p90"]
        tf = "✅" if t90 < SLO_TTFT else "⛔"
        pf = "✅" if tp90 < SLO_TPOT else "⛔"
        row = (f"{c:>5} {s:>4}/{t:<4} "
               f"{p['ttft_ms']['p50']:>9.0f} {t90:>9.0f} {p['ttft_ms']['p99']:>9.0f} "
               f"{p['tpot_ms']['p50']:>8.1f} {tp90:>8.1f} {p['tpot_ms']['p99']:>8.1f} "
               f"{p['output_tokens_per_s']:>9.1f} {tf:>6} {pf:>6}")
        print(row)
        if t90 < SLO_TTFT and tp90 < SLO_TPOT:
            good_c, good_tp = c, p["output_tokens_per_s"]

    print("-" * 130)
    if good_c:
        print(f"  ✅ SLO可用并发 c={good_c}  goodput={good_tp:.1f} tok/s")
    else:
        print(f"  ❌ 全点越线")
    return {"deployment": deployment, "scenario": scenario,
            "slo_concurrency": good_c, "goodput": good_tp, "points": points}

def summary_table(results):
    """Print SLO conclusion summary."""
    print(f"\n{'='*120}")
    print(f"  SLO 结论总表（TTFT p90<{SLO_TTFT}ms / TPOT p90<{SLO_TPOT}ms）")
    print(f"{'='*120}")
    hdr = f"{'部署':<18} {'场景':<8} {'可用并发':>8} {'goodput':>10} {'卡数':>5} {'首个越线(指标+值@c)':>30}"
    print(hdr)
    print("-" * 120)
    for dep, scs in sorted(results.items()):
        cards = 8 if dep == "single_mtp" else 16
        for sc, info in scs.items():
            gpc = info.get("slo_concurrency", "—")
            gp = f"{info['goodput']:.1f}" if info.get("goodput") else "—"
            first = "—"
            for p in info.get("points", []):
                t90, tp90 = p["ttft_ms"]["p90"], p["tpot_ms"]["p90"]
                if t90 >= SLO_TTFT and tp90 >= SLO_TPOT:
                    first = f"TTFT+TPOT c={p['concurrency']}"
                    break
                elif t90 >= SLO_TTFT:
                    first = f"TTFT={t90:.0f}ms @c={p['concurrency']}"
                    break
                elif tp90 >= SLO_TPOT:
                    first = f"TPOT={tp90:.1f}ms @c={p['concurrency']}"
                    break
            print(f"{dep:<18} {sc:<8} {str(gpc):>8} {gp:>10} {cards:>5} {first:>30}")
    print("-" * 120)


def main():
    data = load_all()
    if not data:
        print("No SLO scan results found in", RESULTS_ROOT, flush=True)
        return 1

    results = {}
    for dep in sorted(data):
        results[dep] = {}
        for sc in sorted(data[dep]):
            info = climb_table(dep, sc, data[dep][sc])
            results[dep][sc] = info

    summary_table(results)

    # ── Three core questions ──
    print(f"\n{'='*120}")
    print(f"  核心问题")
    print(f"{'='*120}")

    # Q(a)
    if "pd_nomtp" in data and "16k1k" in data["pd_nomtp"]:
        pts = data["pd_nomtp"]["16k1k"]
        if pts:
            tp90 = pts[0]["tpot_ms"]["p90"]
            c = pts[0]["concurrency"]
            print(f"\n(a) PD+NoMTP 是否全线不可用？")
            print(f"    c={c} TPOT p90={tp90:.1f}ms {'≥' if tp90>=SLO_TPOT else '<'}{SLO_TPOT}ms")

    # Q(b)
    print(f"\n(b) SLO 口径下各部署对比 @16K→1K：")
    for dep in ["pd_mtp", "single_mtp", "dual_mtp"]:
        if dep in data and "16k1k" in data[dep]:
            info = results[dep]["16k1k"]
            gpc = info.get("slo_concurrency", "—")
            gp = info.get("goodput", "—")
            cards = 8 if dep == "single_mtp" else 16
            print(f"    {dep:<15} SLO并发={gpc}  goodput={gp}  {cards}卡")

    # Q(c)
    dps = ["pd_mtp", "single_mtp"]
    if all(d in data and "16k1k" in data[d] for d in dps):
        print(f"\n(c) PD decode 隔离是否让 TPOT 更稳？")
        pd_pts = {p["concurrency"]: p for p in data["pd_mtp"]["16k1k"]}
        s_pts  = {p["concurrency"]: p for p in data["single_mtp"]["16k1k"]}
        for c in sorted(set(pd_pts) & set(s_pts)):
            d = pd_pts[c]["tpot_ms"]["p90"] - s_pts[c]["tpot_ms"]["p90"]
            flag = "✅ PD更稳" if d < 0 else ("=" if abs(d) < 0.5 else "PD更差")
            print(f"    c={c:4d}: PD p90={pd_pts[c]['tpot_ms']['p90']:.1f}ms  "
                  f"单服={s_pts[c]['tpot_ms']['p90']:.1f}ms  Δ={d:+.1f}ms {flag}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
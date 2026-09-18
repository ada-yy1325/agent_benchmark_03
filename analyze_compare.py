"""
GSM8K Detailed Comparison Analyzer
===================================
Compares FP16 vs W8A8 outputs question-by-question.
Usage: python3 analyze_compare.py <fp16_results.json> <w8a8_results.json>
"""

import json
import sys


def analyze(fp16_file, w8a8_file):
    fp16 = json.load(open(fp16_file))
    w8a8 = json.load(open(w8a8_file))

    fp16_results = fp16["results"]
    w8a8_results = w8a8["results"]

    fp16_map = {r["q_idx"]: r for r in fp16_results}
    w8a8_map = {r["q_idx"]: r for r in w8a8_results}

    common_qs = sorted(set(fp16_map.keys()) & set(w8a8_map.keys()))
    total = len(common_qs)
    print(f"Common questions: {total}")

    # ── 1. Config comparison ──
    print("\n" + "=" * 70)
    print("CONFIG COMPARISON")
    print("=" * 70)
    config_keys = ["model", "api_url", "few_shot", "score", "correct",
print(f"{'Item':<20} {'FP16':<25} {'W8A8':<25}")
    print("-" * 70)
    for k in sorted(set(list(fp16.keys()) + list(w8a8.keys()))):
        if k == "results":
            continue
        fv = str(fp16.get(k, "N/A"))
        wv = str(w8a8.get(k, "N/A"))
        match = "✅" if fv == wv else "⚠️"
        print(f"{k:<20} {fv:<25} {wv:<25} {match}")

    # ── 2. Per-question agreement analysis ──
    print("\n" + "=" * 70)
    print("QUESTION-BY-QUESTION AGREEMENT")
    print("=" * 70)

    both_correct = 0
    both_wrong = 0
    fp16_only = 0
    w8a8_only = 0
    disagreements = []

    for q_idx in common_qs:
        fp = fp16_map[q_idx]
        wp = w8a8_map[q_idx]
        fp_c = fp["correct"]
        w8_c = wp["correct"]

        if fp_c and w8_c:
            both_correct += 1
        elif not fp_c and not w8_c:
            both_wrong += 1
        elif fp_c and not w8_c:
            fp16_only += 1
            disagreements.append({
                "q_idx": q_idx, "winner": "FP16",
                "true": fp["true_answer"],
                "fp16_pred": fp["pred_answer"],
                "w8a8_pred": wp["pred_answer"],
                "fp16_out": fp.get("output_snippet", "")[:200],
                "w8a8_out": wp.get("output_snippet", "")[:200],
            })
        else:
            w8a8_only += 1
            disagreements.append({
                "q_idx": q_idx, "winner": "W8A8",
                "true": fp["true_answer"],
                "fp16_pred": fp["pred_answer"],
                "w8a8_pred": wp["pred_answer"],
                "fp16_out": fp.get("output_snippet", "")[:200],
                "w8a8_out": wp.get("output_snippet", "")[:200],
            })

    print(f"{'':<10} {'Count':<10} {'% of total':<12}")
    print("-" * 32)
    print(f"{'Both ✓':<10} {both_correct:<10} {both_correct / total * 100:<12.2f}")
    print(f"{'Both ✗':<10} {both_wrong:<10} {both_wrong / total * 100:<12.2f}")
    print(f"{'FP16 only ✓':<10} {fp16_only:<10} {fp16_only / total * 100:<12.2f}")
    print(f"{'W8A8 only ✓':<10} {w8a8_only:<10} {w8a8_only / total * 100:<12.2f}")

    agreement = (both_correct + both_wrong) / total * 100
    print(f"\nAgreement rate: {agreement:.2f}%")
# ── 3. McNemar's test ──
    print("\n" + "=" * 70)
    print("McNEMAR'S TEST (Statistical Significance)")
    print("=" * 70)
    b = fp16_only  # FP16 correct, W8A8 wrong
    c = w8a8_only  # W8A8 correct, FP16 wrong
    n_d = b + c
    if n_d > 0:
        chi2 = (abs(b - c) - 1) ** 2 / n_d
        print(f"  FP16 only ✓ (b): {b}")
        print(f"  W8A8 only ✓ (c): {c}")
        print(f"  McNemar χ² = {chi2:.4f}")
        print(f"  χ² > 3.84 → p < 0.05")
        print(f"  → {'⚠️ SIGNIFICANT' if chi2 > 3.84 else '✅ NOT significant'}")

    # ── 4. Finish reason ──
    print("\n" + "=" * 70)
    print("FINISH REASON ANALYSIS")
    print("=" * 70)
    fp16_finish = {}
    w8a8_finish = {}
    for r in fp16_results:
        fr = r.get("finish_reason", "N/A")
        fp16_finish[fr] = fp16_finish.get(fr, 0) + 1
    for r in w8a8_results:
        fr = r.get("finish_reason", "N/A")
        w8a8_finish[fr] = w8a8_finish.get(fr, 0) + 1
    print(f"{'Finish Reason':<20} {'FP16':<10} {'W8A8':<10}")
    print("-" * 40)
    for r in sorted(set(list(fp16_finish.keys()) + list(w8a8_finish.keys()))):
        print(f"{r:<20} {fp16_finish.get(r, 0):<10} {w8a8_finish.get(r, 0):<10}")

    # ── 5. Sample disagreements (first 20) ──
    print("\n" + "=" * 70)
    print("SAMPLE DISAGREEMENTS (first 20)")
    print("=" * 70)
    disagreements.sort(key=lambda x: x["q_idx"])
    for d in disagreements[:20]:
        print(f"\n--- Q{d['q_idx']} (true={d['true']}, winner={d['winner']}) ---")
        print(f"  FP16: {d['fp16_pred']}  W8A8: {d['w8a8_pred']}")
        print(f"  FP16 out: {d['fp16_out'][:120]}")
        print(f"  W8A8 out: {d['w8a8_out'][:120]}")

    # ── 6. Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total compared:           {total}")
    print(f"  Both correct:             {both_correct} ({both_correct/total*100:.2f}%)")
    print(f"  Both wrong:               {both_wrong} ({both_wrong/total*100:.2f}%)")
    print(f"  FP16 only correct:        {fp16_only} ({fp16_only/total*100:.2f}%)")
    print(f"  W8A8 only correct:        {w8a8_only} ({w8a8_only/total*100:.2f}%)")
    print(f"  Disagreement rate:        {n_d/total*100:.2f}%")
    print(f"  Net advantage:            {'W8A8' if c > b else 'FP16'} by {abs(c-b)} qs ({abs(c-b)/total*100:.2f}pp)")
    print(f"  McNemar χ² = {chi2:.4f} {'⚠️' if chi2 > 3.84 else '✅'}")
    print(f"  Agreement rate:           {agreement:.2f}%")

    return {"total": total, "both_correct": both_correct, "both_wrong": both_wrong,
            "fp16_only": fp16_only, "w8a8_only": w8a8_only, "agreement": agreement}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 analyze_compare.py <fp16_results.json> <w8a8_results.json>")
        sys.exit(1)
    analyze(sys.argv[1], sys.argv[2])
                   "n_processed", "errors"]
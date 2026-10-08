#!/usr/bin/env python3
"""
Save DSV4 GPQA-Diamond evaluation summary.
Run after eval_gpqa_dsv4.py completes.
"""
import json
import os
import glob

OUTPUTS_DIR = "./outputs"

def find_latest_output():
    dirs = sorted([d for d in os.listdir(OUTPUTS_DIR) if os.path.isdir(os.path.join(OUTPUTS_DIR, d))])
    return os.path.join(OUTPUTS_DIR, dirs[-1]) if dirs else None

def main():
    latest = find_latest_output()
    if not latest:
        print("❌ No output directory found")
        return

    model_name = "dsv4"
    reviews_dir = os.path.join(latest, "reviews", model_name)
    preds_dir = os.path.join(latest, "predictions", model_name)

    # Load reviews
    correct = 0
    total = 0
    review_files = glob.glob(os.path.join(reviews_dir, "*.jsonl"))
    if review_files:
        with open(review_files[0]) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                ss = d.get("sample_score", {}).get("score", {}).get("value", {})
                acc = ss.get("acc")
                if acc is not None:
                    total += 1
                    if acc == 1.0:
                        correct += 1

    # Load predictions for format check
    format_ok = 0
    pred_files = glob.glob(os.path.join(preds_dir, "*.jsonl"))
    if pred_files:
        with open(pred_files[0]) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = json.loads(line)
                msgs = p.get("messages", [])
                if msgs:
                    last = msgs[-1] if isinstance(msgs, list) else msgs
                    resp = last.get("response", "") or last.get("content", "") or ""
                    if "ANSWER:" in resp[-50:]:
                        format_ok += 1

    accuracy = correct / total * 100 if total > 0 else 0
    format_rate = format_ok / total * 100 if total > 0 else 0

    summary = {
        "model": model_name,
        "dataset": "gpqa_diamond",
        "num_questions": total,
        "correct": correct,
        "accuracy": round(accuracy, 2),
        "format_compliance": f"{format_ok}/{total} ({format_rate:.1f}%)",
        "official_score": 88.17,
        "gap": round(accuracy - 88.17, 2),
    }

    print(f"\n{'='*60}")
    print(f"  DeepSeek-V4-Flash-w8a8-mtp — GPQA-Diamond Results")
    print(f"{'='*60}")
    print(f"  Questions:    {total}")
    print(f"  Correct:      {correct}")
    print(f"  Accuracy:     {accuracy:.2f}%")
    print(f"  Format OK:    {format_ok}/{total} ({format_rate:.1f}%)")
    print(f"  {'─'*40}")
    print(f"  Official:     88.17%")
    print(f"  Gap:          {summary['gap']:+.2f}%")
    print(f"  {'─'*40}")
    print(f"  Time:         See perf table above")
    print(f"{'='*60}\n")

    with open("dsv4_gpqa_result.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ Summary saved to dsv4_gpqa_result.json")

if __name__ == "__main__":
    main()
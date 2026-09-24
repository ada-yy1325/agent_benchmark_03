#!/usr/bin/env python3
"""Quick script to parse gpqa_diamond results from evalscope output."""
import json, sys, os

results_dir = sys.argv[1] if len(sys.argv) > 1 else "./outputs"
if not os.path.isdir(results_dir):
    print(f"Directory not found: {results_dir}")
    sys.exit(1)

dirs = sorted([d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))])
if not dirs:
    print("No output dirs found")
    sys.exit(1)

latest = os.path.join(results_dir, dirs[-1])
print(f"Latest output: {latest}")

# Find review files
for root, dirs2, files in os.walk(latest):
    for f in files:
        if f.endswith(".jsonl") and "review" in root:
            fpath = os.path.join(root, f)
            print(f"\nReview file: {fpath}")
            correct = 0
            total = 0
            with open(fpath) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        score_info = rec.get("sample_score", {})
                        score_val = score_info.get("score", {})
                        acc = score_val.get("value", {}).get("acc")
                        if acc is not None:
                            total += 1
                            if acc == 1.0:
                                correct += 1
                    except (json.JSONDecodeError, KeyError):
                        pass
            if total > 0:
                print(f"  Accuracy: {correct}/{total} = {correct/total*100:.2f}%")
            else:
                print(f"  No scored samples found (total={total})")
                print(f"  File has {sum(1 for _ in open(fpath) if _.strip())} non-empty lines")
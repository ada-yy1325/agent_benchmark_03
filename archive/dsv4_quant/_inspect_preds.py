#!/usr/bin/env python3
"""Detailed per-sample analysis of GPQA-Diamond predictions vs correct answers for W8A8."""
import json, sys, os

rev_file = None
for d in sorted(os.listdir("outputs"), reverse=True):
    candidate = os.path.join("outputs", d, "reviews", "Llama-3.1-8B-Instruct-W8A8", "gpqa_diamond_default.jsonl")
    if os.path.exists(candidate):
        rev_file = candidate
        break

if not rev_file:
    print("No W8A8 review file found!")
    sys.exit(1)

print(f"Review file: {rev_file}")
print()

with open(rev_file) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        rec = json.loads(line)
        ss = rec.get("sample_score", {})
        score = ss.get("score", {})
        acc = score.get("value", {}).get("acc", "N/A")
        extracted = score.get("extracted_prediction", "N/A")
        full_pred = score.get("prediction", "")
        meta = rec.get("sample_metadata", {})
        correct_ans = meta.get("correct_answer", "N/A")
        incorrect = meta.get("incorrect_answers", [])
        target = rec.get("target", "?")
        sid = rec.get("sample_id", "?")

        # Find the ANSWER: line in full prediction
        ans_pos = full_pred.rfind("ANSWER:")
        ans_line = full_pred[ans_pos:ans_pos+80] if ans_pos >= 0 else "NO 'ANSWER:' FOUND"

        print(f"{'='*65}")
        print(f"Sample {sid}:")
        print(f"  Extracted → '{extracted}'  |  Accuracy = {acc}")
        print(f"  Target: {target}")

        if len(incorrect) == 4:
            opts = ["A", "B", "C", "D"]
            for j, (opt_letter, ans_text) in enumerate(zip(opts[:4], [incorrect[0], incorrect[1], incorrect[2], correct_ans])):
                marker = " ← CORRECT" if opt_letter == target else ""
                predicted_marker = " ← EXTRACTED" if opt_letter == extracted else ""
                print(f"    {opt_letter}) {ans_text[:80]}{marker}{predicted_marker}")
        else:
            print(f"  Correct: {correct_ans[:100]}")
            print(f"  Incorrect options: {len(incorrect)}")
            for j, ia in enumerate(incorrect):
                print(f"    {j}) {ia[:100]}")

        print(f"  ANSWER line: {ans_line}")
        print()
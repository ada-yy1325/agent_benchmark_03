#!/usr/bin/env python3
"""Analyze GPQA predictions to find error patterns and truncation issues."""
import json
import sys
import os

def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    preds_file = os.path.join(base_dir, "outputs/20261008_150013/predictions/dsv4/gpqa_diamond_default.jsonl")
    reviews_file = os.path.join(base_dir, "outputs/20261008_150013/reviews/dsv4/gpqa_diamond_default.jsonl")

    predictions = []
    with open(preds_file) as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    reviews = []
    with open(reviews_file) as f:
        for line in f:
            if line.strip():
                reviews.append(json.loads(line))

    print(f"Loaded {len(predictions)} predictions, {len(reviews)} reviews")

    # Analyze predictions
    correct, wrong, format_errors = [], [], []
    wrong_format_ok = 0
    truncated_wrong = 0
    wrong_short = 0

    for i, (pred, review) in enumerate(zip(predictions, reviews)):
        score = review.get("sample_score", {}).get("score", {}).get("value", {}).get("acc", 0)
        msgs = pred.get("messages", [])
        last = msgs[-1] if msgs else {}
        resp = last.get("response", "") or last.get("content", "") or ""
        # Check dalle_0 field for perf metrics
        dalle_0 = last.get("dalle_0", {})
        had_truncation = dalle_0.get("had_truncation", False)
        input_tokens = dalle_0.get("input_tokens", 0)
        output_tokens = dalle_0.get("output_tokens", 0)
        
        has_format = False
        # Check for ANSWER: in the last 100 chars
        if resp and "ANSWER:" in resp[-100:]:
            has_format = True

        entry = {
            "idx": i,
            "response": resp,
            "has_format": has_format,
            "truncated": had_truncation,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

        if score == 1.0:
            correct.append(entry)
        else:
            wrong.append(entry)
            if has_format:
                wrong_format_ok += 1
            else:
                format_errors.append(entry)
            if had_truncation:
                truncated_wrong += 1
            if resp and len(resp.strip()) < 50 and not has_format:
                wrong_short += 1

    print(f"\n{'='*60}")
    print(f"  CORRECT: {len(correct)} / {len(predictions)} ({len(correct)/len(predictions)*100:.2f}%)")
    print(f"  WRONG:   {len(wrong)} / {len(predictions)}")
    print(f"{'='*60}")
    print(f"\n  Wrong answers:")
    print(f"    Format OK (has ANSWER:): {wrong_format_ok}/{len(wrong)} ({wrong_format_ok/len(wrong)*100:.1f}%)")
    print(f"    Format ERROR:           {len(format_errors)}/{len(wrong)}")
    print(f"    Had truncation:         {truncated_wrong}/{len(wrong)}")
    print(f"    Very short (<50 chars): {wrong_short}/{len(wrong)}")

    # Check output token distribution for wrong answers
    if wrong:
        outputs = [w["output_tokens"] for w in wrong if w["output_tokens"] > 0]
        if outputs:
            avg_out = sum(outputs) / len(outputs)
            max_out = max(outputs)
            min_out = min(outputs)
            short_out = sum(1 for o in outputs if o < 50)
            print(f"\n  Output token stats (wrong answers):")
            print(f"    Avg: {avg_out:.0f}, Min: {min_out}, Max: {max_out}")
            print(f"    Very short (<50 tok): {short_out}/{len(outputs)}")

        # Correct answers output tokens
        correct_outputs = [c["output_tokens"] for c in correct if c["output_tokens"] > 0]
        if correct_outputs:
            avg_corr = sum(correct_outputs) / len(correct_outputs)
            print(f"  Output token stats (correct answers):")
            print(f"    Avg: {avg_corr:.0f}")

    # Print sample wrong answers
    print(f"\n{'='*60}")
    print(f"  SAMPLE WRONG ANSWERS WITH FORMAT OK")
    print(f"{'='*60}")
    n = 0
    for w in wrong:
        if w["has_format"] and n < 10:
            resp = w["response"]
            print(f"\n--- Wrong #{w['idx']} | out_tok={w['output_tokens']} trunc={w['truncated']} ---")
            # Print the last 300 chars to see the answer format
            print(f"  Last 300 chars: {resp[-300:]}")
            n += 1

    print(f"\n{'='*60}")
    print(f"  SAMPLE WRONG ANSWERS WITH FORMAT ERROR")
    print(f"{'='*60}")
    n = 0
    for fe in format_errors[:10]:
        resp = fe["response"]
        print(f"\n--- Wrong #{fe['idx']} | out_tok={fe['output_tokens']} trunc={fe['truncated']} ---")
        print(f"  Response (first 300): {resp[:300]}")
        print(f"  Response (last 200):  {resp[-200:]}")
        n += 1

    # Check for reasoning/thinking content
    print(f"\n{'='*60}")
    print(f"  THINKING/REASONING DETECTION")
    print(f"{'='*60}")
    for tag, group in [("correct", correct), ("wrong", wrong)]:
        has_thinking = sum(1 for g in group if g["response"] and "" in g["response"])
        has_think_token = sum(1 for g in group if g["response"] and " thinking" in g["response"])
        print(f"  {tag}: ' response' in {has_thinking}/{len(group)}, ' thinking' in {has_think_token}/{len(group)}")

if __name__ == "__main__":
    main()
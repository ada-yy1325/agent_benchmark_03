#!/usr/bin/env python3
"""Inspect GPQA-Diamond predictions to see what the model actually answered."""
import json, sys

pred_file = sys.argv[1] if len(sys.argv) > 1 else None
if not pred_file:
    # auto-detect
    import os, glob
    preds = sorted(glob.glob("outputs/*/predictions/*/gpqa_diamond_default.jsonl"))
    if not preds:
        print("No prediction files found")
        sys.exit(1)
    pred_file = preds[-1]

print(f"Reading: {pred_file}")
with open(pred_file) as f:
    lines = [l.strip() for l in f if l.strip()]

print(f"Total predictions: {len(lines)}")

for i, line in enumerate(lines[:5]):
    rec = json.loads(line)
    msgs = rec.get("messages", [])
    
    # Find user question and assistant answer
    question = ""
    answer = ""
    for msg in msgs:
        content = msg.get("content", "")
        role = msg.get("role", "")
        if role == "user" and not question:
            question = content[:300]
        if role == "assistant" and not answer:
            answer = content
    
    # Also check perf metrics
    perf = msg.get("perf_metrics", {}) if msgs else {}
    
    print(f"\n{'='*60}")
    print(f"Sample {i+1}:")
    print(f"  Question [{len(question)} chars]: {question[:200]}...")
    print(f"  Answer length: {len(answer)} chars")
    last_part = answer[-600:] if len(answer) > 600 else answer
    print(f"  Last 600 chars:")
    print(f"  '''{last_part}'''")
    print()
    
    # Get score from review file
    rev_file = pred_file.replace("predictions", "reviews")
    if os.path.exists(rev_file):
        with open(rev_file) as rf:
            for rline in rf:
                rline = rline.strip()
                if not rline: continue
                rrec = json.loads(rline)
                rid = rrec.get("custom_fields", {}).get("origin_id", "")
                if rid and rid == rec.get("custom_fields", {}).get("origin_id", ""):
                    score_info = rrec.get("sample_score", {})
                    score_val = score_info.get("score", {})
                    acc = score_val.get("value", {}).get("acc")
                    if acc is not None:
                        print(f"  Score: {acc}")
                    break
    
    if perf:
        latency = perf.get("latency", 0)
        ttft = perf.get("ttft", 0)
        tpot = perf.get("tpot", 0)
        print(f"  Latency: {latency:.1f}s, TTFT: {ttft*1000:.0f}ms, TPOT: {tpot*1000:.0f}ms")
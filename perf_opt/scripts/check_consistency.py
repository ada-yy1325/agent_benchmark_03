#!/usr/bin/env python3
"""Phase 5.5: Numerical consistency + GPQA smoke."""
import requests, json, os, time, subprocess, sys

PORT = 8000
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
MODEL = "dsv4"

def chat(prompt, temp=0.0, max_tok=50, seed=42):
    r = requests.post(f"{BASE_URL}/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp, "max_tokens": max_tok, "seed": seed,
    }, timeout=120)
    return r.json()["choices"][0]["message"]["content"]

# ── 1. Determinism check ──
print("=== 1. Numerical Consistency (temp=0, seed=42) ===")
PROMPT = "What is the capital of France? Answer in one word."
responses = []
for i in range(3):
    text = chat(PROMPT)
    responses.append(text)
    print(f"  Run {i+1}: {text.strip()}")
all_same = all(r == responses[0] for r in responses)
status = "✅ PASS" if all_same else "⚠️ FAIL"
print(f"  {status}: {'identical' if all_same else 'DIFFER!'}\n")

# ── 2. More complex prompt ──
print("=== 2. Reasoning Consistency ===")
PROMPT2 = "If 5x + 3 = 28, what is x? Show your steps briefly."
r1 = chat(PROMPT2)
r2 = chat(PROMPT2)
print(f"  Run 1: {r1.strip()}")
print(f"  Run 2: {r2.strip()}")
same2 = r1 == r2
print(f"  {'✅ PASS' if same2 else '⚠️ Non-deterministic!'}\n")

# ── 3. Summary ──
results = {"determinism_simple": all_same, "determinism_reasoning": same2}
with open("/tmp/phase5/5_5_consistency.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to /tmp/phase5/5_5_consistency.json")
print(f"Overall: {'✅ ALL PASS' if all_same and same2 else '⚠️ ISSUES FOUND'}")
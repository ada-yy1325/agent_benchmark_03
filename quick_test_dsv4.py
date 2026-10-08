#!/usr/bin/env python3
"""
Quick 5-query smoke test for DeepSeek-V4-Flash-w8a8-mtp.
Verifies the vLLM server is responding correctly before running full evaluation.
"""
import requests
import sys
import time

API_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "dsv4"

test_questions = [
    "What is the capital of France?",
    "Explain quantum computing in one sentence.",
    "What is 2+2?",
    "Who wrote Romeo and Juliet?",
    "What is the chemical symbol for water?",
]

passed = 0
failed = 0

for i, q in enumerate(test_questions):
    print(f"\n[{i+1}/5] Q: {q}", flush=True)
    t0 = time.time()
    try:
        resp = requests.post(
            API_URL,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": q}],
                "max_tokens": 256,
                "temperature": 0,
            },
            timeout=180,
        )
        elapsed = time.time() - t0
        if resp.status_code == 200:
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            print(f"  A: {answer[:200]}", flush=True)
            print(f"  ⏱ {elapsed:.1f}s | tokens: {usage.get('total_tokens', '?')}", flush=True)
            print(f"  ✅ OK", flush=True)
            passed += 1
        else:
            print(f"  ❌ Error {resp.status_code}: {resp.text[:300]}", flush=True)
            failed += 1
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ❌ Exception after {elapsed:.1f}s: {e}", flush=True)
        failed += 1

print(f"\n{'='*50}", flush=True)
print(f"  Smoke test: {passed}/{passed+failed} passed", flush=True)
if passed == 5:
    print(f"  ✅ All good — ready for full evaluation!", flush=True)
else:
    print(f"  ⚠️  {failed} failures — check server log", flush=True)
print(f"{'='*50}", flush=True)
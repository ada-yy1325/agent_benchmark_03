#!/usr/bin/env python3
"""Test completions API with token IDs (same as bench_serving.py)."""
import requests, json
from transformers import AutoTokenizer

model_dir = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-self"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# Tokenize a real prompt
prompt_text = "Paris is the capital of"
prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
print(f"Prompt text: {prompt_text}")
print(f"Token count: {len(prompt_ids)}")

url = "http://127.0.0.1:8000/v1/completions"
payload = {
    "model": "dsv4",
    "prompt": prompt_ids,  # Send as list of ints
    "temperature": 0.0,
    "max_tokens": 20,
    "seed": 42,
}
r = requests.post(url, json=payload, timeout=120)
d = r.json()
text = d["choices"][0]["text"]
print("Completion:", repr(text))
print("Decoded:", tokenizer.decode(tokenizer.encode(prompt_text, add_special_tokens=False) + tokenizer.encode(text, add_special_tokens=False))[len(prompt_text):])
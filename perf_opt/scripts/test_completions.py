#!/usr/bin/env python3
"""Test completions API with longer real prompt to see if model generates any non-dash output."""
import requests, json
from transformers import AutoTokenizer

model_dir = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-self"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# Try a long, well-structured prompt
prompt_text = """The following is a conversation between a human and an AI assistant. The assistant is helpful, harmless, and honest.

Human: What is the capital of France? Answer in one word.

Assistant: The capital of France is"""
prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
print(f"Token count: {len(prompt_ids)}")

url = "http://127.0.0.1:8000/v1/completions"
payload = {
    "model": "dsv4",
    "prompt": prompt_ids,
    "temperature": 0.0,
    "max_tokens": 30,
    "seed": 42,
}
r = requests.post(url, json=payload, timeout=120)
d = r.json()
text = d["choices"][0]["text"]
decoded = tokenizer.decode(tokenizer.encode(prompt_text, add_special_tokens=False) + tokenizer.encode(text, add_special_tokens=False))[len(prompt_text):]
print("Decoded completion:", repr(decoded))
print("Raw text:", repr(text))
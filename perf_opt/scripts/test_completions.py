#!/usr/bin/env python3
"""Quick completions API test."""
import requests, json

url = "http://127.0.0.1:8000/v1/completions"
payload = {
    "model": "dsv4",
    "prompt": "Paris is the capital of",
    "temperature": 0.0,
    "max_tokens": 20,
    "seed": 42,
}
r = requests.post(url, json=payload, timeout=120)
d = r.json()
text = d["choices"][0]["text"]
print("Completions output:", repr(text))
#!/usr/bin/env python3
"""Test ModelScope download URL and speed."""
from modelscope.hub.api import HubApi
import requests, time, sys

api = HubApi()
files = api.get_model_files("Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp", recursive=True)
safetensors = [f for f in files if f.get("Path","").endswith(".safetensors")]

if not safetensors:
    print("No safetensor files found!")
    sys.exit(1)

path = safetensors[0].get("Path")
size = safetensors[0].get("Size", 0)
print(f"First file: {path}, size: {size/1e9:.1f} GB")

# Try direct download URL
url = f"https://modelscope.cn/api/v1/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp/repo?Revision=master&FilePath={path}"
r = requests.head(url, allow_redirects=True)
print(f"HEAD status: {r.status_code}")
print(f"Final URL: {r.url[:200]}")
print(f"Content-Length: {r.headers.get('content-length', '?')}")

# Also try the raw URL pattern
url2 = f"https://modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp/resolve/master/{path}"
r2 = requests.head(url2, allow_redirects=True)
print(f"\nAlt URL status: {r2.status_code}")
print(f"Alt Final URL: {r2.url[:200]}")
print(f"Alt Content-Length: {r2.headers.get('content-length', '?')}")

# Speed test: download first 10MB
print(f"\nSpeed test (downloading 10MB from {url[:80]}...):")
start = time.time()
r3 = requests.get(url, stream=True, timeout=30, headers={"Range": "bytes=0-10485759"})
downloaded = 0
for chunk in r3.iter_content(chunk_size=65536):
    downloaded += len(chunk)
    if downloaded >= 10*1024*1024:
        break
elapsed = time.time() - start
speed = downloaded / elapsed / 1024 / 1024
print(f"Downloaded {downloaded/1024/1024:.1f} MB in {elapsed:.1f}s = {speed:.2f} MB/s")
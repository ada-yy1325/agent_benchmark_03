#!/usr/bin/env python3
"""Check DeepSeek-V4-Flash W8A8 model info on ModelScope."""
from modelscope.hub.api import HubApi
api = HubApi()

# Check non-MTP version
try:
    info = api.get_model("Eco-Tech/DeepSeek-V4-Flash-w8a8")
    print(f"Non-MTP version exists! Size: {info.get('file_size', 'unknown')}")
except Exception as e:
    print(f"No non-MTP version: {e}")

# Check MTP version config.json
files = api.get_model_files("Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp")
print(f"\nTotal files in MTP repo: {len(files)}")

file_sizes = {}
for f in files:
    name = f.get("Name", "")
    size = f.get("Size", 0)
    if name.endswith(".safetensors"):
        file_sizes[name] = size

print(f"Safetensor files: {len(file_sizes)}")
total_gb = sum(file_sizes.values()) / 1e9
print(f"Total model size: {total_gb:.1f} GB")

# Show some filenames
for name in sorted(file_sizes.keys())[:5]:
    print(f"  {name}: {file_sizes[name]/1e9:.1f} GB")

# Try to download config.json
import requests, json
import os

url = "https://modelscope.cn/api/v1/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp/repo?Revision=master&FilePath=config.json"
r = requests.get(url)
if r.status_code == 200:
    config = r.json()
    print(f"\nConfig: model_type={config.get('model_type')}, architectures={config.get('architectures')}")
    print(f"  quantization_config: {config.get('quantization_config', 'N/A')}")
else:
    print(f"\nFailed to get config.json: {r.status_code}")

# Also check the non-MTP model's config if exists
try:
    url2 = "https://modelscope.cn/api/v1/models/Eco-Tech/DeepSeek-V4-Flash-w8a8/repo?Revision=master&FilePath=config.json"
    r2 = requests.get(url2)
    if r2.status_code == 200:
        config2 = r2.json()
        print(f"\nNon-MTP Config: model_type={config2.get('model_type')}, architectures={config2.get('architectures')}")
except:
    pass
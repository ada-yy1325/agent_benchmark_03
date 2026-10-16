#!/usr/bin/env python3
"""Download GLM-5.3-Flash-W8A8 from ModelScope."""
from modelscope.hub.snapshot_download import snapshot_download
import os

target = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/GLM-5.3-Flash-W8A8"
os.makedirs(target, exist_ok=True)

print("Starting download...")
result = snapshot_download("Eco-Tech/GLM-5.3-Flash-w8a8", local_dir=target, max_workers=8)
print(f"Download complete: {result}")
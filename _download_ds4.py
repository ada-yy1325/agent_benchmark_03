#!/usr/bin/env python3
"""Download DS V4 W8A8 with ModelScope snapshot_download."""
from modelscope.hub.snapshot_download import snapshot_download
import time

target = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-mtp"
print(f"Downloading to: {target}")
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
t0 = time.time()

result = snapshot_download("Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp", local_dir=target, max_workers=16)

elapsed = time.time() - t0
print(f"Done! Elapsed: {elapsed/60:.1f} min")
print(f"Result: {result}")
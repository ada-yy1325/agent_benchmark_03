#!/usr/bin/env python3
"""Test aria2c download from ModelScope."""
import subprocess, os
from modelscope.hub.api import HubApi

api = HubApi()
files = api.get_model_files("Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp", recursive=True)

# Find optional file (small)
for f in files:
    path = f.get("Path", "")
    if "optional/" in path:
        url = f"https://modelscope.cn/api/v1/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp/repo?Revision=master&FilePath={path}"
        print(f"Testing: {path}")
        print(f"URL length: {len(url)}")
        
        # aria2c test - download first 10MB to check speed
        local_path = "/tmp/test_ds4_part"
        cmd = [
            "aria2c", "-x", "8", "-s", "8", "-k", "1M",
            "--max-tries=2", "--retry-wait=2",
            "--continue=true", "--no-conf=true",
            "--max-download-limit=0",
            "--summary-interval=5",
            "-d", "/tmp",
            "-o", "test_ds4_part",
            url
        ]
        print(f"Running: {' '.join(cmd[:8])} ... {url[-40:]}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        print(result.stdout[-1000:] if result.stdout else "")
        print(result.stderr[-500:] if result.stderr else "")
        
        # Check speed from output
        for line in (result.stdout or "").split("\n"):
            if "MiB" in line and ("DL" in line or "speed" in line.lower()):
                print(f"SPEED: {line}")
        
        # Clean up
        if os.path.exists("/tmp/test_ds4_part"):
            sz = os.path.getsize("/tmp/test_ds4_part")
            os.remove("/tmp/test_ds4_part")
            print(f"Downloaded {sz/1024/1024:.1f} MB")
        break
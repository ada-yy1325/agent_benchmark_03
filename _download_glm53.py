#!/usr/bin/env python3
"""Check GLM-5.3-Flash model info and start download."""
import sys, os, time

model_id = "zai-org/GLM-5.3-Flash"
local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "GLM-5.3-Flash")

# Check if already exists
if os.path.exists(local_dir):
    files = os.listdir(local_dir)
    if any(f.endswith(".safetensors") or f.endswith(".bin") or f.endswith(".pt") for f in files):
        print(f"Model already exists at {local_dir}, {len(files)} files")
        # Count total size
        total = 0
        for root, dirs, files in os.walk(local_dir):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        print(f"Total size: {total/1024**3:.1f} GB")
        sys.exit(0)

print(f"Downloading {model_id} to {local_dir}")
print(f"Estimated size: ~640GB, this will take a long time...")

# Use modelscope with multi-threaded download
from modelscope.hub.snapshot_download import snapshot_download

start = time.time()
result = snapshot_download(
    model_id,
    local_dir=local_dir,
    resume_download=True,
)
elapsed = time.time() - start
print(f"Download completed in {elapsed/60:.1f} minutes")
print(f"Saved to: {result}")

# Count files and size
total = 0
for root, dirs, files in os.walk(local_dir):
    for f in files:
        total += os.path.getsize(os.path.join(root, f))
print(f"Total size: {total/1024**3:.1f} GB")
print(f"Total files: {len(os.listdir(local_dir))}")
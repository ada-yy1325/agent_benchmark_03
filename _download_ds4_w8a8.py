#!/usr/bin/env python3
"""Download DeepSeek-V4-Flash-w8a8-mtp using aria2c for maximum speed."""
from modelscope.hub.api import HubApi
import subprocess, os, sys, json, concurrent.futures, time

target = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-mtp"
os.makedirs(target, exist_ok=True)

# Get file listing
api = HubApi()
all_files = api.get_model_files("Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp", recursive=True)
print(f"Total files: {len(all_files)}")

# Download all files in parallel using aria2c
def download_file(f):
    path = f.get("Path", f.get("Name", ""))
    size = f.get("Size", 0)
    local_path = os.path.join(target, path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Construct download URL
    # ModelScope raw file URL pattern
    url = f"https://modelscope.cn/api/v1/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp/repo?Revision=master&FilePath={path}"
    
    cmd = [
        "aria2c", "-x", "8", "-s", "8", "-k", "1M",
        "--max-tries=5", "--retry-wait=5",
        "--continue=true", "--no-conf=true",
        "-d", os.path.dirname(local_path),
        "-o", os.path.basename(local_path) + ".tmp",
        url
    ]
    
    temp_path = local_path + ".tmp"
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            os.rename(temp_path, local_path)
            return (path, "OK", size)
        else:
            return (path, f"FAILED({result.returncode})", 0)
    except Exception as e:
        return (path, f"ERROR({e})", 0)

# First download small files (config, tokenizer, etc.)
print("\nDownloading config files first...")
small_files = [f for f in all_files if not f.get("Path","").endswith(".safetensors")]
big_files = [f for f in all_files if f.get("Path","").endswith(".safetensors")]

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    futures = [ex.submit(download_file, f) for f in small_files]
    for i, fut in enumerate(concurrent.futures.as_completed(futures)):
        path, status, _ = fut.result()
        print(f"  [{i+1}/{len(small_files)}] {os.path.basename(path)}: {status}")

# Now download safetensor files in parallel
print(f"\nDownloading {len(big_files)} model weight files...")
start = time.time()
completed_bytes = 0
total_bytes = sum(f.get("Size", 0) for f in big_files)

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(download_file, f): f for f in big_files}
    for i, fut in enumerate(concurrent.futures.as_completed(futures)):
        path, status, size = fut.result()
        completed_bytes += size
        elapsed = time.time() - start
        speed = completed_bytes / elapsed / 1024 / 1024 if elapsed > 0 else 0
        pct = completed_bytes / total_bytes * 100 if total_bytes > 0 else 0
        print(f"  [{i+1}/{len(big_files)}] {os.path.basename(path)[:50]}: {status} ({pct:.1f}%, {speed:.1f} MB/s)")
        
print(f"\nDownload complete! Total: {completed_bytes/1e9:.1f} GB in {time.time()-start:.0f}s")
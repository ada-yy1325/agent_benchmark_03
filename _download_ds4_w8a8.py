#!/usr/bin/env python3
"""Download DeepSeek-V4-Flash-w8a8-mtp using aria2c batch mode for max speed."""
from modelscope.hub.api import HubApi
import subprocess, os, time

target = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-mtp"
os.makedirs(target, exist_ok=True)

api = HubApi()
all_files = api.get_model_files("Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp", recursive=True)
print(f"Total files: {len(all_files)}")

batch_file = "/tmp/aria2c_batch.txt"
urls = []
total_size = 0
dirs_created = set()

for f in all_files:
    path = f.get("Path", "")
    if not path or path.endswith("/"):
        continue
    local_path = os.path.join(target, path)
    d = os.path.dirname(local_path)
    if d not in dirs_created:
        os.makedirs(d, exist_ok=True)
        dirs_created.add(d)
    url = f"https://modelscope.cn/api/v1/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp/repo?Revision=master&FilePath={path}"
    size = f.get("Size", 0)
    total_size += size
    urls.append((url, local_path, path[-50:], size))

with open(batch_file, "w") as f:
    for url, local_path, _, _ in urls:
        f.write(f"{url}\n")
        f.write(f"  out={os.path.basename(local_path)}\n")
        f.write(f"  dir={os.path.dirname(local_path)}\n")

print(f"Files: {len(urls)}, total: {total_size/1e9:.1f} GB")

cmd = [
    "aria2c",
    "--max-concurrent-downloads=8",
    "--max-connection-per-server=8",
    "--split=8",
    "--min-split-size=1M",
    "--continue=true",
    "--max-tries=5",
    "--retry-wait=5",
    "--console-log-level=notice",
    "--summary-interval=10",
    "--no-conf=true",
    "-i", batch_file
]

print(f"Starting... {len(urls)} files, 8 parallel x 8 connections")
print(f"{total_size/1e9:.1f} GB total")
t0 = time.time()
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    line = line.strip()
    if line:
        elapsed = time.time() - t0
        print(f"[{elapsed/60:.1f}m] {line}")
proc.wait()
print(f"Done! {time.time()-t0:.0f}s, exit={proc.returncode}")
os.remove(batch_file)
#!/usr/bin/env python3
"""Monitor download progress of DS V4 Flash w8a8."""
import subprocess, os, json, time

model_dir = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-mtp"
log_file = "/tmp/ds4_dl.log"
status_file = "/tmp/ds4_status.txt"

# Get total file count from original listing
try:
    with open("/tmp/aria2c_batch.txt") as f:
        urls = [l.strip() for l in f if l.strip() and not l.startswith("  ") and not l.startswith("#")]
    total_urls = len(urls)
except:
    total_urls = "?"

alive = subprocess.run(["pgrep", "-af", "aria2c.*batch"], capture_output=True, text=True).returncode == 0

# Count complete safetensors (no .aria2 sidecar)
safetensors = []
partial = []
if os.path.isdir(model_dir):
    for fn in os.listdir(model_dir):
        if fn.endswith(".safetensors.aria2"):
            partial.append(fn.replace(".aria2",""))
        elif fn.endswith(".safetensors"):
            safetensors.append(fn)
n_complete = len([s for s in safetensors if s not in set(partial)])

# Get real size (not apparent = preallocated)
r = subprocess.run(["du", "-sb", model_dir], capture_output=True, text=True)
try:
    real_bytes = int(r.stdout.split()[0])
except:
    real_bytes = 0

# Get latest speed from log
speed_line = ""
try:
    with open(log_file) as f:
        for line in f:
            if "DL:" in line and "CN:" in line:
                speed_line = line.strip()
except:
    pass

elapsed = ""
try:
    with open(log_file) as f:
        first = f.readline()
        if "[" in first and "m]" in first:
            pass
        for line in f:
            if "[" in line and "m]" in line[:10]:
                elapsed = line[:line.index("]")+1] if "]" in line else ""
except:
    pass

print(f"{'='*50}")
print(f"  DS V4 Flash w8a8 下载进度")
print(f"{'='*50}")
print(f"  ⏱  运行时间    : {elapsed}")
print(f"  📦 完整文件    : {n_complete} / {total_urls}")
print(f"  🔄 下载中      : {len(partial)} 个")
print(f"  💾 真实大小    : {real_bytes/1e9:.1f} GB")
print(f"  🚀 进程状态    : {'✅ 运行中' if alive else '❌ 已结束'}")
print(f"  📊 最新速度    : {speed_line[-100:] if speed_line else 'N/A'}")
print(f"{'='*50}")

# Save for quick look
with open(status_file, "w") as f:
    data = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "complete": n_complete,
        "total": total_urls,
        "partial": len(partial),
        "size_gb": round(real_bytes/1e9, 1),
        "alive": alive,
        "timestamp": time.time()
    }
    json.dump(data, f)

if not alive:
    print("⚠️ 下载进程已结束！请检查是否完成或出错。")
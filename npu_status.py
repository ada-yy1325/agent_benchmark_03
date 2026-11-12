#!/usr/bin/env python3
"""Check NPU status and write to status file."""
import subprocess, sys, os

out = os.path.join(os.path.dirname(__file__), "npu_status.txt")
lines = []

# Check NPU
try:
    import torch
    lines.append(f"NPU available: {torch.npu.is_available()}")
    lines.append(f"NPU count: {torch.npu.device_count()}")
except Exception as e:
    lines.append(f"torch error: {e}")

# Check devices
r = subprocess.run("ls /dev/davinci* 2>&1", shell=True, capture_output=True, text=True)
lines.append(f"devices: {r.stdout.strip() or r.stderr.strip()}")

# Check ports
for port in [8000, 8001]:
    r = subprocess.run(f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{port}/health 2>&1",
                       shell=True, capture_output=True, text=True)
    lines.append(f"port {port} health: {r.stdout.strip()}")

# Check tmux
r = subprocess.run("tmux ls 2>&1", shell=True, capture_output=True, text=True)
lines.append(f"tmux: {r.stdout.strip() or r.stderr.strip()}")

with open(out, "w") as f:
    f.write("\n".join(lines))
print("status written to", out)
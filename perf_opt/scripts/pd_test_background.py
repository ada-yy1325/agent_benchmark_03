#!/usr/bin/env python3
"""Background test: wait for P, then test PD proxy vs direct D."""
import json, time, subprocess, sys
from datetime import datetime

LOG = "/tmp/pd_proxy_test_results.log"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def curl(url, payload, timeout=120):
    """Run curl, return (http_code, total_time, body)."""
    cmd = [
        "curl", "-s", "-o", "/dev/stdout",
        "-w", "\n%{http_code} %{time_total}",
        "--max-time", str(timeout),
        url,
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload, separators=(",", ":")),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
        out = r.stdout.strip()
        if "\n" in out:
            body, meta = out.rsplit("\n", 1)
        else:
            body, meta = out, "0 0"
        parts = meta.split()
        code = int(parts[0]) if parts else 0
        t = float(parts[1]) if len(parts) > 1 else 0.0
        return code, t, body
    except Exception as e:
        log(f"curl error: {e}")
        return 0, 0.0, ""

def check_health(port):
    """Check if a server is healthy."""
    try:
        r = subprocess.run(
            ["curl", "-s", "-m", "5", f"http://127.0.0.1:{port}/health"],
            capture_output=True, text=True, timeout=10,
        )
        return "ok" in r.stdout.lower()
    except Exception:
        return False

def get_metrics(port, pattern):
    """Get metrics matching pattern."""
    try:
        r = subprocess.run(
            ["curl", "-s", f"http://127.0.0.1:{port}/metrics"],
            capture_output=True, text=True, timeout=10,
        )
        lines = [l for l in r.stdout.split("\n") if pattern in l]
        return "\n".join(lines)
    except Exception:
        return ""

# ── Main ──
log("=== PD Proxy Background Test ===")

# Step 0: Wait for P
log("Waiting for P:8000...")
for i in range(60):
    if check_health(8000):
        log(f"P is UP! (attempt {i+1})")
        break
    log(f"  attempt {i+1}/60: P not ready, sleep 15s")
    time.sleep(15)
else:
    log("P did NOT start after 15 minutes. Aborting.")
    sys.exit(1)

# Wait a bit more for model to fully load
time.sleep(10)

# Step 1: Baseline - direct to D
log("=== Test 1: Direct to D (baseline) ===")
prompt = "The capital of France is"
code, t, body = curl(
    "http://127.0.0.1:8001/v1/completions",
    {"model": "dsv4", "prompt": prompt, "max_tokens": 20}
)
log(f"Direct D: HTTP {code} | total={t:.2f}s")
try:
    j = json.loads(body)
    ttft = j.get("usage", {}).get("completion_tokens", "?")
    log(f"  Response: {j.get('choices',[{}])[0].get('text','')[:80]}")
except Exception:
    log(f"  Body: {body[:200]}")

# Step 2: Via proxy P→D
log("=== Test 2: Via proxy P→D ===")
code, t, body = curl(
    "http://127.0.0.1:8002/v1/completions",
    {"model": "dsv4", "prompt": prompt, "max_tokens": 20}
)
log(f"Proxy P→D: HTTP {code} | total={t:.2f}s")
try:
    j = json.loads(body)
    log(f"  Response: {j.get('choices',[{}])[0].get('text','')[:80]}")
except Exception:
    log(f"  Body: {body[:200]}")

# Step 3: Run benchmark via proxy (5 requests)
log("=== Test 3: Proxy benchmark (5 requests) ===")
times = []
for i in range(5):
    code, t, body = curl(
        "http://127.0.0.1:8002/v1/completions",
        {"model": "dsv4", "prompt": f"Tell me about France in {10+i} words.", "max_tokens": 50}
    )
    log(f"  Request {i+1}: HTTP {code} | total={t:.2f}s")
    if 200 == code:
        times.append(t)

if times:
    avg = sum(times) / len(times)
    log(f"Proxy avg: {avg:.2f}s over {len(times)} requests")

# Step 4: Metrics
log("=== Test 4: Metrics ===")
for name, port in [("D", 8001), ("P", 8000)]:
    m = get_metrics(port, "prompt_tokens_by_source")
    log(f"\n{name} metrics (port {port}):\n{m or '(none)'}")

# Step 5: Proxy log
log("=== Proxy log tail ===")
try:
    with open("/tmp/pd_proxy.log") as f:
        lines = f.readlines()
    for line in lines[-40:]:
        log(f"  PROXY: {line.rstrip()}")
except Exception as e:
    log(f"  Error reading proxy log: {e}")

log("=== DONE ===")
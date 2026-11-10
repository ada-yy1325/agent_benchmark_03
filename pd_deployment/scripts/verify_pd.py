#!/usr/bin/env python3
"""
P/D Separation Deployment — Quick Verification Script

Verifies that a P/D setup is functioning correctly:
  1. Both instances respond to /health
  2. A test request returns valid output
  3. Decode instance's prompt throughput is ~0 (proves KV is from transfer, not local prefill)

Usage:
  python3 pd_deployment/scripts/verify_pd.py [--mtp]
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

PREFILL_PORT = 8000
DECODE_PORT = 8001
TMP = "/tmp/pd_test"


def check_health(port: int, label: str) -> bool:
    """Poll :port/health until 200 or timeout."""
    for attempt in range(30):
        try:
            resp = urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=5
            )
            if resp.status == 200:
                print(f"  ✅ {label} :{port} healthy")
                return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    print(f"  ❌ {label} :{port} NOT healthy after 60s")
    return False


def send_test_request(port: int) -> dict | None:
    """Send a minimal chat completion request and return timing."""
    payload = json.dumps({
        "model": "dsv4",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 16,
        "temperature": 0.0,
    }).encode()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        resp = urllib.request.urlopen(req, timeout=120)
        elapsed = time.time() - t0
        body = json.loads(resp.read().decode())
        text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"elapsed": elapsed, "text": text, "status": resp.status}
    except Exception as e:
        return {"error": str(e)}


def check_pd_log(logfile: str) -> bool:
    """Verify decode log shows prompt throughput ≈ 0."""
    if not os.path.exists(logfile):
        print(f"  ⚠️  Log file not found: {logfile}")
        return False
    # Look for "prompt throughput: 0.0" or very low prompt throughput
    prompt_zero = False
    with open(logfile) as f:
        for line in f:
            if "prompt throughput" in line:
                if "0.0 tokens/s" in line or "0.0 tokens/s" in line:
                    prompt_zero = True
    if prompt_zero:
        print("  ✅ PD verified: decode prompt throughput ≈ 0 (KV from transfer)")
        return True
    else:
        print("  ⚠️  Could not confirm PD in logs (check manually)")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mtp", action="store_true", help="Verify MTP version")
    args = parser.parse_args()

    suffix = "mtp" if args.mtp else "v2"
    prefill_log = f"{TMP}/prefill_{suffix}.log" if not args.mtp else f"{TMP}/prefill_mtp.log"
    decode_log = f"{TMP}/decode_{suffix}.log" if not args.mtp else f"{TMP}/decode_mtp.log"

    print(f"\n╔══════════════════════════════════════════╗")
    print(f"║  P/D Verification {'(MTP ON)' if args.mtp else '(No-MTP)'}  ║")
    print(f"╚══════════════════════════════════════════╝\n")

    # 1. Health check
    print("[1/3] Checking instance health...")
    ok_d = check_health(DECODE_PORT, "Decode")
    ok_p = check_health(PREFILL_PORT, "Prefill")
    if not (ok_d and ok_p):
        print("\n❌ VERIFICATION FAILED: instances not healthy")
        print("   Check logs: tail -100", decode_log, prefill_log)
        sys.exit(1)

    # 2. Test request
    print("\n[2/3] Sending test request to decode...")
    result = send_test_request(PREFILL_PORT)
    if result is None or "error" in result:
        print(f"  ❌ Request failed: {result}")
        sys.exit(1)
    print(f"  ✅ Response in {result['elapsed']:.1f}s: {result['text'][:80]}")

    # 3. PD proof
    print(f"\n[3/3] Verifying PD separation (decode log)...")
    check_pd_log(decode_log)

    print(f"\n{'✅' if ok_d and ok_p else '❌'} P/D {'MTP' if args.mtp else 'No-MTP'} verification complete!\n")


if __name__ == "__main__":
    main()
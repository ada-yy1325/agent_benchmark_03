#!/usr/bin/env python3
"""
Two-phase PD test client:
  Phase 1: Send to producer → prefill + save KV to pool
  Phase 2: Send SAME prompt to consumer → load KV → decode

Validates AscendStoreConnector KV pool sharing between two instances.
Consumer's TTFT should be dramatically lower (skips prefill).
"""
import time, json, urllib.request, sys, os

os.environ.setdefault("OMP_PROC_BIND", "false")

PRODUCER_URL = "http://127.0.0.1:8000/v1/chat/completions"
CONSUMER_URL = "http://127.0.0.1:8001/v1/chat/completions"
MODEL_NAME = "dsv4"
MAX_TOKENS = 512
REQUESTS_URL = f"http://127.0.0.1:8000/requests"
REQUESTS_URL2 = f"http://127.0.0.1:8001/requests"

PROMPTS = [
    "用中文写一篇500字关于人工智能对教育影响的文章。",
    "Explain quantum computing in 300 words to a high school student.",
    "写一个Python函数实现快速排序算法，并给出详细注释。",
    ("什么是PD分离？在LLM推理中，prefill和decode分离有什么好处？"
     "请详细解释架构设计和优缺点。"),
    "给我讲一个关于AI和人类友谊的短篇科幻故事。",
]


def post_json(url, data, timeout=300):
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    response = b""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            # For streaming: accumulate all chunks
            while True:
                chunk = r.read(8192)
                if not chunk:
                    break
                response += chunk
        elapsed = time.time() - t0
        text = response.decode("utf-8", errors="replace")
        return {"elapsed": elapsed, "raw": text, "success": True}
    except Exception as e:
        elapsed = time.time() - t0
        return {"elapsed": elapsed, "error": str(e), "success": False}


def get_metrics(url):
    """Fetch vLLM /metrics for queue info."""
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def main():
    print("=" * 60)
    print("PD Two-Phase KV Sharing Test")
    print(f"Producer: {PRODUCER_URL}")
    print(f"Consumer: {CONSUMER_URL}")
    print("=" * 60)

    payload_template = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "PLACEHOLDER"}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": False,
    }

    results = []
    for i, prompt in enumerate(PROMPTS):
        print(f"\n--- Request {i+1}/{len(PROMPTS)} ---")
        print(f"Prompt: {prompt[:60]}...")

        payload = dict(payload_template)
        payload["messages"] = [{"role": "user", "content": prompt}]

        # Phase 1: Producer
        print("  [1/2] Producer (prefill)...")
        r1 = post_json(PRODUCER_URL, payload)
        if r1["success"]:
            print(f"  ✓ Producer: {r1['elapsed']:.2f}s")
            # Parse first token time from response
            try:
                data = json.loads(r1["raw"])
                tokens_used = data.get("usage", {}).get("completion_tokens", 0)
                print(f"    completion_tokens={tokens_used}")
            except Exception:
                pass
        else:
            print(f"  ✗ Producer FAIL: {r1.get('error', 'unknown')}")
            continue

        time.sleep(0.5)

        # Phase 2: Consumer
        print("  [2/2] Consumer (load KV)...")
        r2 = post_json(CONSUMER_URL, payload)
        if r2["success"]:
            ttft = _extract_ttft(r2["raw"])
            print(f"  ✓ Consumer: {r2['elapsed']:.2f}s (TTFT≈{ttft:.2f}s)")
        else:
            print(f"  ✗ Consumer FAIL: {r2.get('error', 'unknown')}")

        results.append({
            "prompt": prompt[:60],
            "producer_time": r1["elapsed"],
            "consumer_time": r2["elapsed"] if r2["success"] else 0,
            "producer_ok": r1["success"],
            "consumer_ok": r2["success"],
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        speedup = r["producer_time"] / r["consumer_time"] if r["consumer_time"] > 0 else 0
        print(f"  {r['prompt'][:40]:40s} prod={r['producer_time']:.1f}s "
              f"cons={r['consumer_time']:.1f}s speedup={speedup:.1f}x")
    print("=" * 60)


def _extract_ttft(raw_text: str) -> float:
    """Try to extract time to first token from streaming or metadata."""
    try:
        data = json.loads(raw_text)
        # Try to find prepfill_wall_time or ttft metadata
        return data.get("prefill_wall_time", data.get("metrics", {}).get("ttft", 0))
    except Exception:
        return 0.0


if __name__ == "__main__":
    main()
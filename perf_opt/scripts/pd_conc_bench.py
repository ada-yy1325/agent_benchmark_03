#!/usr/bin/env python3
"""
PD Concurrent Benchmark Client (v2, 2026-11-06).
Single-stage concurrent requests to D (8001).
PD KV cache transfer is handled internally by Mooncake connector.
Usage:
  python3 pd_conc_bench.py --num-requests 64 --concurrency 32 --input-chars 5400 --output-tokens 200
"""
import argparse, json, time, sys, statistics, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

def _send_request(prompt, max_tokens, timeout=300):
    """Send one completion request, read streaming response to capture TTFT."""
    payload = {
        "model": "dsv4",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    req = urllib.request.Request(
        D_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft_ms = -1
    total_tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            # Parse SSE streaming response
            buf = b""
            while True:
                chunk = r.read(65536)
                if not chunk:
                    break
                buf += chunk
                # Process complete lines
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    lstr = line.decode("utf-8", errors="replace").strip()
                    if not lstr or lstr.startswith(":"):
                        continue
                    if lstr.startswith("data: "):
                        data_str = lstr[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            obj = json.loads(data_str)
                            choices = obj.get("choices", [])
                            if choices and ttft_ms < 0:
                                ttft_ms = (time.perf_counter() - t0) * 1000
                            total_tokens += 1
                        except json.JSONDecodeError:
                            pass
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "e2e_ms": elapsed,
            "ttft_ms": ttft_ms if ttft_ms >= 0 else elapsed,
            "out_tokens": total_tokens,
            "ok": True,
        }
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return {"e2e_ms": elapsed, "ttft_ms": -1, "out_tokens": 0,
                "ok": False, "error": str(e)}


def _pct(data, p):
    if not data:
        return -1
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f, c = int(k), k - int(k)
    return s[f] + c * (s[f + 1] - s[f]) if f + 1 < len(s) else s[f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-requests", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--input-chars", type=int, default=5400)
    ap.add_argument("--output-tokens", type=int, default=200)
    args = ap.parse_args()

    prompt_word = "The quick brown fox " * (args.input_chars // 20)
    prompt_base = ("Please analyze:\n\n" + prompt_word)[:args.input_chars]
    tasks = [{
        "idx": i,
        "id": f"r{i}_{uuid.uuid4().hex[:6]}",
        "prompt": prompt_base + f" [req_{i}]",
    } for i in range(args.num_requests)]

    print(f"[bench] PD v2 — {args.num_requests} req @ c{args.concurrency} "
          f"→ D:8001, prompt~{args.input_chars}ch, max_tokens={args.output_tokens}")

    t0 = time.perf_counter()
    results = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(_send_request, t["prompt"], args.output_tokens): t["idx"]
            for t in tasks
        }
        for f in as_completed(futures):
            r = f.result()
            idx = futures[f]
            results[idx] = r

    wall = time.perf_counter() - t0

    ttft_vals = [r["ttft_ms"] for r in results.values() if r["ok"]]
    e2e_vals = [r["e2e_ms"] for r in results.values() if r["ok"]]
    total_out = sum(r["out_tokens"] for r in results.values() if r["ok"])
    fail = sum(1 for r in results.values() if not r["ok"])

    if fail:
        for i, r in results.items():
            if not r["ok"]:
                print(f"[bench] FAIL r{i}: {r.get('error','?')}")

    throughput = total_out / wall if wall > 0 else 0

    print(f"\n{'='*60}")
    print(f"RESULTS  c={args.concurrency}  n={args.num_requests}  "
          f"prompt~{args.input_chars}ch")
    print(f"{'='*60}")
    print(f"  TTFT p50      : {_pct(ttft_vals, 50):.0f}ms  "
          f"p99: {_pct(ttft_vals, 99):.0f}ms")
    print(f"  E2E  p50      : {_pct(e2e_vals, 50):.0f}ms  "
          f"p99: {_pct(e2e_vals, 99):.0f}ms")
    print(f"  Wall time     : {wall:.1f}s")
    print(f"  Total output  : {total_out} tokens")
    print(f"  Throughput    : {throughput:.1f} tok/s")
    print(f"  Success       : {len(ttft_vals)}/{args.num_requests}")
    print(f"{'='*60}")

    result = {
        "concurrency": args.concurrency,
        "num_requests": args.num_requests,
        "input_chars": args.input_chars,
        "output_tokens": args.output_tokens,
        "wall_s": wall,
        "ttft_p50_ms": _pct(ttft_vals, 50),
        "ttft_p99_ms": _pct(ttft_vals, 99),
        "e2e_p50_ms": _pct(e2e_vals, 50),
        "e2e_p99_ms": _pct(e2e_vals, 99),
        "total_output_tokens": total_out,
        "throughput_tok_s": throughput,
        "ok": len(ttft_vals),
        "fail": fail,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
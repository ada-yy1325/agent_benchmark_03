#!/usr/bin/env python3
"""
PD Concurrent Benchmark Client (reconstructed 2026-11-06).
Two-stage: Stage1 P prefill pool → Stage2 D decode pool.
Usage: python3 pd_conc_bench.py --num-requests 64 --p-concurrency 32 --d-concurrency 32 --input-chars 5400
"""
import argparse, json, time, sys, statistics, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

P_URL = "http://127.0.0.1:8000/v1/completions"
D_URL = "http://127.0.0.1:8001/v1/completions"
BOOTSTRAP = "http://127.0.0.1:8000"
P_LOG = "/tmp/pd_test/prefill_v2.log"


def _get_engine_id(timeout=120):
    """Extract P engine_id from prefill log.
    vLLM 0.23 removed /query endpoint.
    Engine ID = kv_ip:kv_port (default if not set in config).
    """
    import re, os
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if not os.path.exists(P_LOG):
                time.sleep(2)
                continue
            with open(P_LOG, "r", errors="replace") as f:
                text = f.read()
            ip_m = re.search(r"kv_ip='([^']+)'", text)
            port_m = re.search(r"kv_port=(\d+)", text)
            if ip_m and port_m:
                engine_id = f"{ip_m.group(1)}:{port_m.group(1)}"
                print(f"[bench] extracted engine_id={engine_id}")
                return engine_id
        except Exception as e:
            print(f"[bench] _get_engine_id retry: {e}")
        time.sleep(2)
    raise RuntimeError("Cannot extract engine_id from prefill log")


def _send_prefill(prompt, transfer_id, timeout=300):
    """Send one prefill request to P (max_tokens=1)."""
    payload = {
        "model": "dsv4", "prompt": prompt,
        "max_tokens": 1, "temperature": 0.0, "stream": False,
        "kv_transfer_params": json.dumps({
            "do_remote_decode": True, "transfer_id": transfer_id,
        }),
    }
    t0 = time.perf_counter()
    req = urllib.request.Request(
        P_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            json.loads(r.read())
        elapsed = (time.perf_counter() - t0) * 1000
        return {"transfer_id": transfer_id, "prefill_ms": elapsed, "ok": True}
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return {"transfer_id": transfer_id, "prefill_ms": elapsed, "ok": False, "error": str(e)}
def _send_decode(prompt, transfer_id, engine_id, max_tokens=200, timeout=300):
    """Send one decode request to D. Returns TTFT, output tokens."""
    payload = {
        "model": "dsv4", "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0.0, "stream": False,
        "kv_transfer_params": json.dumps({
            "do_remote_prefill": True,
            "remote_engine_id": engine_id,
            "remote_bootstrap_addr": BOOTSTRAP,
        }),
    }
    t0 = time.perf_counter()
    req = urllib.request.Request(
        D_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read())
        elapsed = (time.perf_counter() - t0) * 1000
        out_tokens = resp.get("usage", {}).get("completion_tokens", 0)
        return {"transfer_id": transfer_id, "e2e_ms": elapsed,
                "out_tokens": out_tokens, "ok": True}
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return {"transfer_id": transfer_id, "e2e_ms": elapsed,
                "out_tokens": 0, "ok": False, "error": str(e)}
def _pct(data, p):
    """Compute p-th percentile (linear interpolation)."""
    if not data:
        return -1
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = k - f
    return s[f] + c * (s[f + 1] - s[f]) if f + 1 < len(s) else s[f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-requests", type=int, default=16)
    ap.add_argument("--p-concurrency", type=int, default=1)
    ap.add_argument("--d-concurrency", type=int, default=1)
    ap.add_argument("--input-chars", type=int, default=5400)
    ap.add_argument("--output-tokens", type=int, default=200)
    args = ap.parse_args()

    prompt_word = "The quick brown fox " * (args.input_chars // 20)
    prompt_base = ("Please analyze:\n\n" + prompt_word)[:args.input_chars]
    tasks = [{
        "idx": i,
        "transfer_id": f"c_{i}_{uuid.uuid4().hex[:8]}",
        "prompt": prompt_base + f" [req_{i}]",
    } for i in range(args.num_requests)]

    print(f"[bench] {args.num_requests} req, P_c={args.p_concurrency}, "
          f"D_c={args.d_concurrency}, prompt~{args.input_chars}ch")
    print("[bench] Fetching engine_id ...")
    engine_id = _get_engine_id()
    print(f"[bench] engine_id={engine_id}")

    # Stage 1: Prefill
    print(f"\n[stage1] Prefill @ P_c={args.p_concurrency}")
    t1 = time.perf_counter()
    pr = {}
    with ThreadPoolExecutor(max_workers=args.p_concurrency) as pool:
        futs = {pool.submit(_send_prefill, t["prompt"], t["transfer_id"]): t["idx"]
                for t in tasks}
        for f in as_completed(futs):
            r = f.result()
            pr[r["transfer_id"]] = r
    w1 = time.perf_counter() - t1
    pt = [r["prefill_ms"] for r in pr.values() if r["ok"]]
    pf = sum(1 for r in pr.values() if not r["ok"])
    pp50 = statistics.median(pt) if pt else -1
    # Print first 3 failures for debugging
    fails = [r for r in pr.values() if not r["ok"]]
    for f in fails[:3]:
        print(f"[stage1] FAIL id={f['transfer_id']}: {f.get('error','?')}")
    print(f"[stage1] wall={w1:.1f}s, ok={len(pt)}, fail={pf}, "
          f"prefill_p50={pp50:.0f}ms")

    # Stage 2: Decode
    print(f"\n[stage2] Decode @ D_c={args.d_concurrency}")
    t2 = time.perf_counter()
    dr = {}
    with ThreadPoolExecutor(max_workers=args.d_concurrency) as pool:
        futs = {pool.submit(_send_decode, t["prompt"], t["transfer_id"],
                            engine_id, args.output_tokens): t["idx"]
                for t in tasks}
        for f in as_completed(futs):
            r = f.result()
            dr[r["transfer_id"]] = r
    w2 = time.perf_counter() - t2
    et = [r["e2e_ms"] for r in dr.values() if r["ok"]]
    to = sum(r["out_tokens"] for r in dr.values() if r["ok"])
    df = sum(1 for r in dr.values() if not r["ok"])
    # Print first 3 decode failures
    dfails = [r for r in dr.values() if not r["ok"]]
    for f in dfails[:3]:
        print(f"[stage2] FAIL id={f['transfer_id']}: {f.get('error','?')}")
    ep50 = statistics.median(et) if et else -1
    ep99 = _pct(et, 99) if et else -1
    tp = to / w2 if w2 > 0 else 0

    print(f"\n{'='*60}")
    print(f"RESULTS  c=(P{args.p_concurrency}/D{args.d_concurrency})  "
          f"n={args.num_requests}  prompt~{args.input_chars}ch")
    print(f"{'='*60}")
    print(f"  P prefill p50 : {pp50:.0f}ms  (stage wall={w1:.1f}s, fail={pf})")
    print(f"  D TTFT(E2E)   : p50={ep50:.0f}ms  p99={ep99:.0f}ms")
    print(f"  D stage wall  : {w2:.1f}s  fail={df}")
    print(f"  Total output  : {to} tokens")
    print(f"  Throughput    : {tp:.1f} tok/s")
    print(f"{'='*60}")

    result = {
        "p_concurrency": args.p_concurrency,
        "d_concurrency": args.d_concurrency,
        "num_requests": args.num_requests,
        "input_chars": args.input_chars,
        "output_tokens": args.output_tokens,
        "stage1_wall_s": w1,
        "stage2_wall_s": w2,
        "prefill_p50_ms": pp50,
        "d_e2e_p50_ms": ep50,
        "d_e2e_p99_ms": ep99,
        "total_output_tokens": to,
        "throughput_tok_s": tp,
        "prefill_ok": len(pt),
        "prefill_fail": pf,
        "decode_ok": len(et),
        "decode_fail": df,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
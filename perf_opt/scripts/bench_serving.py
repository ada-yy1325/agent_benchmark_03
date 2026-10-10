#!/usr/bin/env python3
"""
定长压测客户端 for DeepSeek-V4-Flash W8A8 on vLLM.
随机 token id 作为输入，走 POST /v1/completions 流式接口，
采样 NPU 显存，输出标准 JSON。

Usage:
    python3 bench_serving.py \
        --base-url http://127.0.0.1:8000/v1 \
        --served-name dsv4 \
        --model-dir /path/to/model \
        --input-len 4096 \
        --output-len 1024 \
        --concurrency 8 \
        --num-requests 64 \
        --case-id 4k_out1024_c8 \
        --out results/matrix/4k_out1024_c8.json \
        --npu-stats npu_stats/4k_out1024_c8.csv
"""
import argparse
import asyncio
import json
import random
import re
import subprocess
import threading
import time
from datetime import datetime

import httpx
from transformers import AutoTokenizer


def pct(xs, p):
    """Compute the p-th percentile."""
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


class NpuSampler(threading.Thread):
    """Background thread: samples npu-smi info every ~1s."""

    def __init__(self, path: str):
        super().__init__(daemon=True)
        self.path = path
        self.on = True
        self.rows = []

    def run(self):
        while self.on:
            t = time.time()
            try:
                out = subprocess.run(
                    ["npu-smi", "info"],
                    capture_output=True, text=True, timeout=10
                ).stdout
                used = [int(x) for x in re.findall(r"(\d+)\s*/\s*\d+\s*MB", out)]
            except Exception:
                used = []
                out = ""
            self.rows.append((t, used, out))
            time.sleep(1)

    def save(self):
        with open(self.path, "w") as f:
            n_npu = len(self.rows[0][1]) if self.rows else 0
            f.write("timestamp," + ",".join(f"npu{i}_used_mb" for i in range(n_npu)) + "\n")
            for t, used, _ in self.rows:
                f.write(f"{t}," + ",".join(map(str, used)) + "\n")


async def one_request(client, args, tokenizer, prompt_ids, idx):
    """Send one streaming completions request."""
    payload = {
        "model": args.served_name,
        "prompt": prompt_ids,
        "max_tokens": args.output_len,
        "temperature": args.temp,
        "stream": True,
        "ignore_eos": True,
        "seed": args.seed,
    }
    t0 = time.time()
    first_chunk = None
    last_chunk = t0
    parts = []
    err = None
    fallback_used = False

    try:
        async with client.stream("POST", "/completions", json=payload) as resp:
            if resp.status_code != 200:
                # Fallback: decode prompt to text
                fallback_used = True
                text_prompt = tokenizer.decode(prompt_ids)
                payload["prompt"] = text_prompt
                async with client.stream("POST", "/completions", json=payload) as resp2:
                    if resp2.status_code != 200:
                        err = f"HTTP {resp2.status_code}: {await resp2.aread()}"
                    else:
                        async for line in resp2.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                obj = json.loads(data)
                                txt = obj["choices"][0].get("text", "")
                                now = time.time()
                                if txt and first_chunk is None:
                                    first_chunk = now
                                if txt:
                                    parts.append(txt)
                                    last_chunk = now
                            except (json.JSONDecodeError, KeyError):
                                pass
            else:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        txt = obj["choices"][0].get("text", "")
                        now = time.time()
                        if txt and first_chunk is None:
                            first_chunk = now
                        if txt:
                            parts.append(txt)
                            last_chunk = now
                    except (json.JSONDecodeError, KeyError):
                        pass
    except Exception as e:
        err = repr(e)

    return {
        "t0": t0,
        "first": first_chunk,
        "last": last_chunk,
        "text": "".join(parts),
        "err": err,
        "fallback_used": fallback_used,
    }

async def main():
    ap = argparse.ArgumentParser(description="定长压测客户端 for DSV4 W8A8")
    for name, tp in [("base-url", str), ("served-name", str), ("model-dir", str),
                     ("case-id", str), ("out", str), ("npu-stats", str)]:
        ap.add_argument("--" + name, required=True, type=tp)
    for name in ["input-len", "output-len", "concurrency", "num-requests"]:
        ap.add_argument("--" + name, required=True, type=int)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    print(f"[bench] Loading tokenizer from {args.model_dir}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size
    print(f"[bench] vocab_size={vocab_size}", flush=True)
    special_ids = set()
    for attr in ["bos_token_id", "eos_token_id", "pad_token_id"]:
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special_ids.add(tid)
    if hasattr(tokenizer, "added_tokens_decoder"):
        for tid in tokenizer.added_tokens_decoder:
            special_ids.add(tid)
    print(f"[bench] special/added token ids to avoid: {sorted(special_ids)}", flush=True)
    safe_min = 100
    safe_max = vocab_size - 1
    while safe_min in special_ids and safe_min < safe_max:
        safe_min += 1
    print(f"[bench] random id range: [{safe_min}, {safe_max})", flush=True)

    def make_prompt():
        return [random.randint(safe_min, safe_max) for _ in range(args.input_len)]

    queue = asyncio.Queue()
    for i in range(args.num_requests):
        queue.put_nowait((i, make_prompt()))

    sampler = NpuSampler(args.npu_stats)
    sampler.start()
    print(f"[bench] NPU sampler started -> {args.npu_stats}", flush=True)

    records = []
    lock = asyncio.Lock()
    completed = 0
    progress_lock = asyncio.Lock()

    async with httpx.AsyncClient(base_url=args.base_url, timeout=600) as client:
        async def worker(worker_id):
            nonlocal completed
            while True:
                try:
                    idx, prompt_ids = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                result = await one_request(client, args, tokenizer, prompt_ids, idx)
                async with lock:
                    records.append(result)
                async with progress_lock:
                    completed += 1
                    if completed % max(1, args.num_requests // 10) == 0:
                        print(f"[bench] {completed}/{args.num_requests}", flush=True)

        t_start = time.time()
        await asyncio.gather(*[worker(w) for w in range(args.concurrency)])
        wall_time = time.time() - t_start

    sampler.on = False
    time.sleep(1.5)
    sampler.save()
    print(f"[bench] NPU stats saved ({len(sampler.rows)} samples)", flush=True)

    ok = [r for r in records if r["err"] is None and r["first"] is not None]
    out_tokens_list = []
    for r in ok:
        encoded = tokenizer(r["text"], add_special_tokens=False)["input_ids"]
        out_tokens_list.append(len(encoded))
    out_tokens = sum(out_tokens_list)
    in_tokens = args.input_len * len(ok)

    ttfts, tpots, e2es = [], [], []
    for r in ok:
        encoded = tokenizer(r["text"], add_special_tokens=False)["input_ids"]
        n_out = len(encoded)
        ttft = (r["first"] - r["t0"]) * 1000
        tpot = (r["last"] - r["first"]) * 1000 / max(1, n_out - 1) if n_out > 1 else 0.0
        e2e = (r["last"] - r["t0"]) * 1000
        ttfts.append(ttft)
        tpots.append(tpot)
        e2es.append(e2e)

    peak = []
    for _, used, _ in sampler.rows:
        for i, v in enumerate(used):
            if i >= len(peak):
                peak.append(v)
            else:
                peak[i] = max(peak[i], v)

    result = {
        "case_id": args.case_id,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "concurrency": args.concurrency,
        "temperature": args.temp,
        "ignore_eos": True,
        "seed": args.seed,
        "planned_requests": args.num_requests,
        "successful_requests": len(ok),
        "failed_requests": args.num_requests - len(ok),
        "wall_time_s": round(wall_time, 3),
        "total_input_tokens": in_tokens,
        "total_output_tokens": out_tokens,
        "output_tokens_per_s": round(out_tokens / wall_time, 2) if wall_time > 0 else 0.0,
        "total_tokens_per_s": round((in_tokens + out_tokens) / wall_time, 2) if wall_time > 0 else 0.0,
        "ttft_ms": {
            "p50": round(pct(ttfts, 50), 2),
            "p90": round(pct(ttfts, 90), 2),
            "p99": round(pct(ttfts, 99), 2),
            "mean": round(sum(ttfts) / len(ttfts), 2) if ttfts else 0,
        },
        "tpot_ms": {
            "p50": round(pct(tpots, 50), 2),
            "p90": round(pct(tpots, 90), 2),
            "p99": round(pct(tpots, 99), 2),
            "mean": round(sum(tpots) / len(tpots), 2) if tpots else 0,
        },
        "e2e_ms": {
            "p50": round(pct(e2es, 50), 2),
            "p99": round(pct(e2es, 99), 2),
        },
        "success_rate": round(len(ok) / args.num_requests, 4),
        "peak_hbm_mb_per_npu": peak,
        "avg_output_tokens_per_request": round(out_tokens / len(ok), 1) if ok else 0,
        "fallback_to_text": any(r.get("fallback_used") for r in ok),
        "timestamp": datetime.now().isoformat(),
        "errors": [r["err"] for r in records if r["err"]],
    }
    with open(args.out, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[bench] Saved to {args.out}", flush=True)
    print(f"[bench] {args.case_id}: {len(ok)}/{args.num_requests} ok, "
          f"out_tok/s={result['output_tokens_per_s']}, "
          f"TTFT p50={result['ttft_ms']['p50']}ms, "
          f"TPOT p50={result['tpot_ms']['p50']}ms", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
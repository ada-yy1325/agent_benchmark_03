#!/usr/bin/env python3
"""PD pipeline benchmark v2: chat/text prompts (KV pool hash works on text path).

Producer (8000): prefill + KV save, max_tokens=2.
Consumer (8001): same text prompt -> KV pool load -> decode 1024 tokens.
Pipeline: producer runs ahead while consumer decodes (prefill/decode overlap).

Usage: python3 pd_pipeline_bench.py --num-requests 32 --consumer-concurrency 32 --lookahead 4
"""
import argparse, json, time, threading
from concurrent.futures import ThreadPoolExecutor

import httpx

PRODUCER = "http://127.0.0.1:8000/v1/chat/completions"
CONSUMER = "http://127.0.0.1:8001/v1/chat/completions"
MODEL = "dsv4"
TARGET_INPUT_TOKENS = 4000
OUTPUT_LEN = 1024

BASE_TEXT = (
    "人工智能技术在近年来取得了飞速发展，从自然语言处理到计算机视觉，从推荐系统到自动驾驶，"
    "深度学习模型在各个领域都展现出了强大的能力。大语言模型通过海量数据的预训练，"
    "习得了丰富的世界知识和推理能力，在问答、写作、编程等任务上表现优异。"
    "推理服务的性能优化是工程落地的重要环节，涉及吞吐量、延迟、显存占用等多个维度。"
    "投机解码、KV缓存管理、批处理调度、张量并行与专家并行等技术被广泛采用，"
    "以提升单卡与多卡的资源利用率。本文讨论的主题包括：模型量化、预填充与解码分离、"
    "动态批处理、连续批处理、前缀缓存、推测解码、以及分布式推理的负载均衡。"
)


def make_prompt(i, tokenizer):
    """Build a ~4000-token text prompt, unique per request index."""
    header = f"请求编号：{i}。请仔细阅读以下背景材料，然后回答相关问题。\n\n"
    text = header
    while len(tokenizer.encode(text, add_special_tokens=False)) < TARGET_INPUT_TOKENS:
        text += BASE_TEXT + "\n"
    return text


def producer_call(client, text):
    t0 = time.time()
    r = client.post(PRODUCER, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 2, "temperature": 0.0, "seed": 42,
    }, timeout=600)
    dt = time.time() - t0
    if r.status_code != 200:
        return {"ok": False, "ms": dt * 1000, "status": r.status_code, "err": r.text[:150]}
    return {"ok": True, "ms": dt * 1000}


def consumer_call(client, text):
    t0 = time.time()
    ttft = None
    n_tokens = 0
    with client.stream("POST", CONSUMER, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": OUTPUT_LEN, "temperature": 0.0, "seed": 42, "stream": True,
        "stream_options": {"include_usage": True},
    }, timeout=600) as r:
        if r.status_code != 200:
            return {"ok": False, "ms": (time.time() - t0) * 1000,
                    "status": r.status_code, "err": r.read().decode()[:150]}
        for line in r.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            ch = (obj.get("choices") or [{}])[0]
            txt = ch.get("text") or (ch.get("message") or {}).get("content") or ""
            if txt and ttft is None:
                ttft = (time.time() - t0) * 1000
            if obj.get("usage") and obj["usage"].get("completion_tokens"):
                n_tokens = obj["usage"]["completion_tokens"]
    if ttft is None:
        ttft = (time.time() - t0) * 1000
    return {"ok": True, "ttft_ms": ttft, "tokens": n_tokens,
            "total_ms": (time.time() - t0) * 1000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-requests", type=int, default=32)
    ap.add_argument("--consumer-concurrency", type=int, default=32)
    ap.add_argument("--lookahead", type=int, default=4)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-npu16",
        trust_remote_code=True)
    N = args.num_requests
    texts = [make_prompt(i, tokenizer) for i in range(N)]
    print(f"[pd] generated {N} prompts, tokens≈{len(tokenizer.encode(texts[0], add_special_tokens=False))}",
          flush=True)

    t0 = time.time()
    producer_ms = []
    consumer_ttft = []
    consumer_tokens = 0
    failures = []
    lock = threading.Lock()
    prod_idx = 0
    prod_done = threading.Event()
    cons_queue = []

    def produce(client):
        nonlocal prod_idx
        while True:
            with lock:
                if prod_idx >= N:
                    break
                i = prod_idx
                prod_idx += 1
            res = producer_call(client, texts[i])
            with lock:
                producer_ms.append(res["ms"])
                if not res["ok"]:
                    failures.append(f"producer[{i}]: {res}")
                cons_queue.append((i, texts[i]))
            if res["ok"] and (i % 10 == 0):
                print(f"[pd] producer {i}/{N} {res['ms']:.0f}ms", flush=True)

    def consume(client):
        nonlocal consumer_tokens
        while True:
            with lock:
                if not cons_queue and prod_done.is_set():
                    break
                if not cons_queue:
                    time.sleep(0.01)
                    continue
                i, text = cons_queue.pop(0)
            res = consumer_call(client, text)
            with lock:
                if res["ok"]:
                    consumer_ttft.append(res["ttft_ms"])
                    consumer_tokens += res["tokens"]
                else:
                    failures.append(f"consumer[{i}]: {res}")
                if len(consumer_ttft) % 10 == 0:
                    print(f"[pd] consumer done {len(consumer_ttft)} "
                          f"ttft={res.get('ttft_ms', 0):.0f}ms", flush=True)

    with ThreadPoolExecutor(max_workers=args.consumer_concurrency + 1) as ex:
        prod_fut = ex.submit(produce, httpx.Client(timeout=600))
        cons_futs = [ex.submit(consume, httpx.Client(timeout=600))
                     for _ in range(args.consumer_concurrency)]
        prod_fut.result()
        prod_done.set()
        for f in cons_futs:
            f.result()

    wall = time.time() - t0
    out = {
        "num_requests": N,
        "input_tokens_per_prompt": len(tokenizer.encode(texts[0], add_special_tokens=False)),
        "wall_s": round(wall, 1),
        "producer_ms_p50": round(sorted(producer_ms)[len(producer_ms) // 2], 1) if producer_ms else None,
        "consumer_ttft_p50": round(sorted(consumer_ttft)[len(consumer_ttft) // 2], 1) if consumer_ttft else None,
        "consumer_ttft_p99": round(sorted(consumer_ttft)[int(len(consumer_ttft) * 0.99)], 1) if consumer_ttft else None,
        "consumer_tokens": consumer_tokens,
        "output_tok_s": round(consumer_tokens / wall, 1) if wall else None,
        "successful": len(consumer_ttft),
        "failures": failures[:10],
    }
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    with open("/tmp/pd_test/pipeline_result.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("[pd] result -> /tmp/pd_test/pipeline_result.json", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 4 audit: MTP acceptance rate on NATURAL TEXT prompts (vs random tokens).

Cline's 4.73% acceptance was measured on random token-id prompts
(bench_serving.py make_prompt). Natural text may be more predictable,
so this probe re-measures acceptance with realistic prompts.

Run on remote against a running MTP=1 server:
    python3 perf_opt/scripts/probe_accept_natural.py --port 8000 \
        --out perf_opt/results/mtp/accept_natural.json
"""
import argparse, json, time, urllib.request, datetime

PROMPTS = [
    "请解释一下什么是Transformer模型中的注意力机制，包括Q、K、V三个矩阵的作用，以及多头注意力和缩放点积注意力的计算过程。",
    "在深度学习模型部署中，量化是一种常见的优化手段。请详细说明8比特权重量化（W8A8）的原理、常见实现方式，以及对模型精度和推理速度的影响。",
    "Explain the difference between eager execution and graph mode in deep learning frameworks, and discuss how CUDA graphs improve inference latency for autoregressive language models.",
    "大规模语言模型推理服务的性能优化通常涉及多个维度：吞吐量、首字延迟（TTFT）、每输出token延迟（TPOT）。请分析这三者之间的关系，以及在批处理调度中如何权衡。",
    "Write a Python function that reads a JSON file, extracts all numeric fields recursively, and returns their sum. Include error handling for missing files and invalid JSON.",
    "What is speculative decoding in large language models? Describe how a draft model proposes tokens and how the target model verifies them, including the concept of acceptance rate.",
    "请比较数据并行（DP）、张量并行（TP）和专家并行（EP）在大型MoE模型训练与推理中的适用场景，并说明它们各自的通信开销特点。",
    "Explain the concept of KV cache in transformer inference, why it grows linearly with sequence length, and how prefix caching can reduce redundant computation across requests.",
]

MODEL_NAME = "dsv4"


def metrics(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as r:
        text = r.read().decode()
    draft = acc = 0
    for ln in text.split("\n"):
        if "spec_decode_num_draft_tokens_total" in ln and not ln.startswith("#"):
            draft = float(ln.split()[-1])
        if "spec_decode_num_accepted_tokens_total" in ln and not ln.startswith("#"):
            acc = float(ln.split()[-1])
    return draft, acc


def one_request(port, prompt, max_tokens=256, temperature=0.0, seed=42):
    payload = json.dumps({
        "model": MODEL_NAME, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": temperature, "seed": seed,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=payload, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        body = json.loads(r.read().decode())
    ttft = ttft2 = None
    usages = body.get("usage", {})
    return time.time() - t0, usages.get("completion_tokens", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--repeat", type=int, default=2,
                    help="repeat each prompt N times (total = len(PROMPTS)*N)")
    args = ap.parse_args()

    d1, a1 = metrics(args.port)
    print(f"[natural] metrics BEFORE: draft={d1:.0f} accepted={a1:.0f}", flush=True)

    t0 = time.time()
    ok, out_toks = 0, 0
    for rep in range(args.repeat):
        for p in PROMPTS:
            try:
                dur, ntok = one_request(args.port, p)
                ok += 1
                out_toks += ntok
                print(f"[natural] {ok}/{len(PROMPTS)*args.repeat} "
                      f"dur={dur:.1f}s out_tok={ntok}", flush=True)
            except Exception as e:
                print(f"[natural] FAIL: {e}", flush=True)

    d2, a2 = metrics(args.port)
    dd, da = d2 - d1, a2 - a1
    rate = da / dd if dd > 0 else None
    print(f"[natural] metrics AFTER: draft={d2:.0f} accepted={a2:.0f}", flush=True)
    print(f"[natural] ACCEPTANCE: {da:.0f}/{dd:.0f} = "
          f"{rate*100:.2f}%" if rate is not None else "[natural] ACCEPTANCE: N/A",
          flush=True)

    result = {
        "phase": "4-audit-natural-text-acceptance",
        "date": datetime.datetime.now().isoformat(),
        "wall_time_s": round(time.time() - t0, 1),
        "requests_ok": ok,
        "total_output_tokens": out_toks,
        "draft_tokens_delta": dd,
        "accepted_tokens_delta": da,
        "acceptance_rate": rate,
        "note": "natural text prompts, temp=0.0, greedy acceptance semantics; "
                "compare with random-token rate 4.35-4.73%",
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[natural] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

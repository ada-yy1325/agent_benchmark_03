#!/usr/bin/env python3
"""PD ≥16K two-phase validation — KV transfer test with long prompt.

Uses a prompt long enough to pass the 16384-token cache_transfer_granularity gate.
Phase 1: producer (8000) prefills, max_tokens=64 → saves KV to AscendStore pool.
Phase 2: consumer (8001) gets the SAME prompt → should LOAD KV from pool (not recompute).

Success indicators:
  - consumer log shows "prompt throughput" ~0 for this request
  - consumer TTFT much shorter than a full prefill would take
  - consumer log shows actual KV cache hit
"""

import json, os, sys, time, urllib.request, urllib.error

PRODUCER_URL = "http://127.0.0.1:8000/v1/chat/completions"
CONSUMER_URL = "http://127.0.0.1:8001/v1/chat/completions"
MODEL_NAME = "dsv4"
TARGET_TOKENS = 17000  # comfortably above 16384 gate

# Build a prompt that yields ≥TARGET_TOKENS tokens.
# Using a self-contained novel-style paragraph — each Chinese char ≈ 1.5-2 tokens in DSV4.
PARAGRAPH = (
    "在人工智能技术飞速发展的今天，大语言模型已经成为了推动科技进步的核心力量。"
    "从自然语言处理到计算机视觉，从自动驾驶到医疗诊断，AI的应用场景日益广泛。"
    "深度学习算法的突破使得机器能够以前所未有的精度理解和生成人类语言，"
    "这为教育、科研、商业等各个领域带来了革命性的变化。"
    "然而，随着模型规模的不断扩大，算力需求也呈指数级增长，"
    "如何在有限的硬件资源下实现高效的推理和训练成为了亟待解决的关键问题。"
    "量化技术、模型蒸馏、投机解码等优化手段应运而生，"
    "它们在不显著牺牲模型精度的前提下大幅提升了推理速度。"
    "与此同时，分布式计算、流水线并行、张量并行等技术的成熟，"
    "使得千亿参数级别的大模型能够在多卡集群上高效运行。"
    "未来的研究方向将聚焦于如何进一步降低推理成本，"
    "让更多的人和企业能够享受到人工智能带来的便利。"
    "教育公平、医疗普惠、环境保护等全球性挑战，"
    "也将在AI的助力下找到更加创新和可持续的解决方案。"
)

# Repeat the paragraph enough times
# Each paragraph is ~200 Chinese chars → ~300-400 tokens
# Need ~50 repeats for 17000 tokens
repeats = 55
PROMPT = (PARAGRAPH * repeats) + "\n\n请用中文总结以上内容的十个核心观点，每个观点不少于五十字。"

print(f"Prompt length: {len(PROMPT)} chars, estimated {len(PROMPT)//2} tokens (assuming ~2 chars/token)")
print(f"Target: ≥{TARGET_TOKENS} tokens to pass 16384 granularity gate")
print()


def post(url, prompt, max_tokens, timeout=600):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
        elapsed = time.time() - t0
        usage = data.get("usage", {})
        return {
            "ok": True, "elapsed": elapsed,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "text": data.get("choices", [{}])[0].get("message", {}).get("content", "")[:120],
        }
    except urllib.error.HTTPError as e:
        err_body = e.read()[:500].decode("utf-8", errors="replace")
        return {"ok": False, "elapsed": time.time() - t0,
                "error": f"HTTP {e.code}: {err_body}"}
    except Exception as e:
        return {"ok": False, "elapsed": time.time() - t0, "error": str(e)}


def main():
    print("PD ≥16K two-phase validation")
    print("=" * 60)

    # Phase 1: Producer prefill
    print("\n[1/2] Producer prefill (max_tokens=64) ...")
    t0_p = time.time()
    r1 = post(PRODUCER_URL, PROMPT, 64)
    p_elapsed = time.time() - t0_p
    print(f"  producer: ok={r1['ok']} wall={p_elapsed:.1f}s "
          f"pt={r1.get('prompt_tokens', '?')} ct={r1.get('completion_tokens', '?')}")
    if not r1["ok"]:
        print(f"  ❌ PRODUCER FAILED: {r1.get('error')}")
        return 1

    pt = r1.get("prompt_tokens", 0)
    if pt < TARGET_TOKENS:
        print(f"  ⚠️  WARNING: prompt_tokens={pt} < {TARGET_TOKENS}, may not pass granularity gate!")
    else:
        print(f"  ✅ prompt_tokens={pt} ≥ {TARGET_TOKENS}, gate should be passed")

    # Give producer time to flush KV into the AscendStore pool
    flush_wait = 8
    print(f"\n  Waiting {flush_wait}s for KV flush to pool...")
    time.sleep(flush_wait)

    # Phase 2: Consumer load KV
    print("\n[2/2] Consumer load KV (max_tokens=1024) ...")
    t0_c = time.time()
    r2 = post(CONSUMER_URL, PROMPT, 1024)
    c_elapsed = time.time() - t0_c
    print(f"  consumer: ok={r2['ok']} wall={c_elapsed:.1f}s "
          f"pt={r2.get('prompt_tokens', '?')} ct={r2.get('completion_tokens', '?')}")
    if not r2["ok"]:
        print(f"  ❌ CONSUMER FAILED: {r2.get('error')}")
        return 1

    print(f"\n  consumer text: {r2.get('text', '')}")

    # Verdict
    print("\n" + "=" * 60)
    print(f"RESULT:")
    print(f"  Producer: {r1['elapsed']:.1f}s wall, {pt} prompt tokens → KV saved?")
    print(f"  Consumer: {c_elapsed:.1f}s wall")

    if c_elapsed < 30:
        verdict = "PASS ✅ — consumer fast (<30s), likely KV pool HIT"
    elif c_elapsed < 120:
        verdict = "POSSIBLE ⚠️ — consumer {:.0f}s, check log for prompt_throughput ~0".format(c_elapsed)
    else:
        verdict = "MISS ❌ — consumer slow (≥120s), likely local recompute (KV pool miss)"

    print(f"  VERDICT: {verdict}")
    print(f"\n  ALSO CHECK consumer log (tail -50 /tmp/pd_test/decode_v2.log):")
    print(f"    - Look for 'SpecDecoding metrics' with prompt_throughput ~0")
    print(f"    - Look for 'kv_transfer' or 'cache' related log lines")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
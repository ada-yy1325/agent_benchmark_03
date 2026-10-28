#!/usr/bin/env python3
"""PD 4K two-phase validation (full-hash fix check).

Phase 1: producer (8000) prefills a ~4K-token TEXT prompt, max_tokens=64.
Phase 2: consumer (8001) gets the SAME prompt, max_tokens=1024.
If the full-hash patch works, the consumer must load the whole prompt KV from
the AscendStore pool (its log shows prompt throughput ~0 and TTFT < 1s),
instead of only matching the first 512 tokens.
"""
import json, time, urllib.request, urllib.error

PRODUCER_URL = "http://127.0.0.1:8000/v1/chat/completions"
CONSUMER_URL = "http://127.0.0.1:8001/v1/chat/completions"
MODEL_NAME = "dsv4"

# ~4000 tokens of text (Chinese + English mix, repeats a paragraph)
BASE = ("人工智能技术的快速发展正在深刻改变教育行业的形态。从个性化学习路径规划，"
        "到智能辅导系统的普及，再到自动批改作业与生成教学素材，AI已经渗透到教学的每一个环节。"
        "教师可以利用大模型快速生成针对不同学生水平的练习题目，学生则可以通过对话式学习助手随时提问。"
        "与此同时，教育公平问题也迎来了新的解决思路：偏远地区的孩子同样能够享受到高质量的数字教育资源。"
        "当然，技术的引入也伴随着挑战，例如数据隐私保护、算法偏见以及过度依赖技术带来的思维惰性。"
        "如何在效率与人文关怀之间取得平衡，是每一位教育工作者都需要思考的问题。")
PROMPT = (BASE * 22) + " 请总结以上内容的核心观点，并给出三个具体建议。"


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
            "text": data.get("choices", [{}])[0].get("message", {}).get("content", "")[:80],
        }
    except urllib.error.HTTPError as e:
        return {"ok": False, "elapsed": time.time() - t0,
                "error": f"HTTP {e.code}: {e.read()[:200]}"}
    except Exception as e:
        return {"ok": False, "elapsed": time.time() - t0, "error": str(e)}


def main():
    print("PD 4K two-phase validation")
    print(f"prompt chars={len(PROMPT)}")
    print("\n[1/2] Producer prefill (max_tokens=64) ...")
    r1 = post(PRODUCER_URL, PROMPT, 64)
    print(f"  producer: ok={r1['ok']} elapsed={r1.get('elapsed', 0):.1f}s "
          f"pt={r1.get('prompt_tokens', '?')} ct={r1.get('completion_tokens', '?')}")
    if not r1["ok"]:
        print("  producer FAILED:", r1.get("error"))
        return 1

    # give the producer a moment to flush KV into the AscendStore pool
    time.sleep(5)

    print("\n[2/2] Consumer load KV (max_tokens=1024) ...")
    r2 = post(CONSUMER_URL, PROMPT, 1024)
    print(f"  consumer: ok={r2['ok']} elapsed={r2.get('elapsed', 0):.1f}s "
          f"pt={r2.get('prompt_tokens', '?')} ct={r2.get('completion_tokens', '?')}")
    if not r2["ok"]:
        print("  consumer FAILED:", r2.get("error"))
        return 1

    print(f"\n  consumer text: {r2.get('text', '')}")

    pt = r1.get("prompt_tokens", 0)
    # consumer TTFT estimate: elapsed - decode time (approx via tokens)
    c_elapsed = r2.get("elapsed", 0.0)
    print("\n" + "=" * 60)
    print(f"RESULT: producer {r1['elapsed']:.1f}s / consumer {c_elapsed:.1f}s "
          f"(prompt={pt} tokens)")
    print(f"VERDICT: {'PASS - consumer fast (KV hit)' if c_elapsed < 15 else 'CHECK - consumer slow, KV miss?'}")
    print("Also check consumer log for 'prompt throughput' ~0 on this request.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

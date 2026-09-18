"""
GSM8K Corrected Evaluation for Qwen3-4B-Base
==============================================
Uses /v1/completions (not chat) with the official Qwen3 paper methodology:
- 4-shot CoT (from Qwen3 tech report §3.3)
- Answer: / Final answer: format
- temperature=0, greedy decoding

Usage:
  python3 eval_gsm8k_base_correct.py [--api_url <url>] [--max_questions <N>]

Requires: vLLM server running on port 8802 (FP16) or 8803 (W8A8)
"""

import re
import json
import time
import sys
import os
import requests

# ── Config (defaults, overridable via CLI) ──────────────────────────────
API_URL = "http://127.0.0.1:8802/v1/completions"
MODEL_NAME = "Qwen3-4B-Base"
MAX_TOKENS = 2048
TEMPERATURE = 0.0
TOP_P = 1.0
SEED = 42
FEW_SHOT_COUNT = 4
# ── 4-shot GSM8K examples (Qwen3 official paper format) ────────────────
# Format: Question / Answer (CoT reasoning) / Final answer: <number>
FEW_SHOT_EXAMPLES = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted.
Final answer: 6

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.
Final answer: 5

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have total in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.
Final answer: 39

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave 20 - 12 = 8 lollipops to Denny.
Final answer: 8"""


def build_prompt(question: str) -> str:
    """Build the prompt: 4-shot examples + test question."""
    return f"{FEW_SHOT_EXAMPLES}\n\nQuestion: {question}\nAnswer:"


def extract_answer(text: str) -> str | None:
    # Priority 1: "Final answer: X" (official paper format)
    match = re.search(r'Final answer:\s*\$?\s*([+-]?\d+\.?\d*)', text, re.IGNORECASE)
    if match:
        raw = match.group(1)
        return raw.rstrip(".")
    # Priority 2: "The answer is X" (alternative format)
    match = re.search(r'The answer is\s*\$?\s*([+-]?\d+\.?\d*)', text, re.IGNORECASE)
    if match:
        raw = match.group(1)
        return raw.rstrip(".")
    # Priority 3: \boxed{X}
    match = re.search(r'\\?boxed\{\$?\s*([+-]?\d+\.?\d*)\}', text)
    if match:
        return match.group(1)
    # Priority 4: ANSWER: X (fallback)
    match = re.search(r'ANSWER:\s*\$?\s*([+-]?\d+\.?\d*)', text, re.IGNORECASE)
    if match:
        return match.group(1).rstrip(".")
    return None


def call_model(prompt: str, url: str) -> tuple:
    """Returns (text, finish_reason) or (None, None) on failure."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        return choice["text"], choice.get("finish_reason", "unknown")
    except Exception as e:
        print(flush=True, f"  [ERROR] API call failed: {e}")
        return None, None
def main():
    # ── Parse CLI args ──
    max_questions = None
    port = 8802
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--max-questions" and i + 2 < len(sys.argv):
            max_questions = int(sys.argv[i + 2])
        elif arg == "--port" and i + 2 < len(sys.argv):
            port = int(sys.argv[i + 2])

    api_url = f"http://127.0.0.1:{port}/v1/completions"
    scope_label = f" (first {max_questions} questions)" if max_questions else ""

    print(flush=True, f"GSM8K Evaluation (Corrected) — {MODEL_NAME}{scope_label}")
    print(flush=True, f"  API: {api_url}")
    print(flush=True, f"  Few-shot: {FEW_SHOT_COUNT}-shot CoT (Qwen3 official paper §3.3)")
    print(flush=True, f"  Format: /v1/completions (raw prompt, no chat template)\n")

    print(flush=True, "Loading GSM8K dataset...")
    local_parquet = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsm8k_test.parquet")
    try:
        import pyarrow.parquet as pq
        tbl = pq.read_table(local_parquet)
        dataset = [{"question": r["question"], "answer": r["answer"]} for r in tbl.to_pylist()]
    except Exception as e:
        print(flush=True, f"  [WARN] Local parquet load failed: {e}")
        # Fallback: try to download with requests
        print(flush=True, "  Trying to download from HuggingFace Hub...")
        resp = requests.get(
            "https://huggingface.co/datasets/gsm8k/resolve/main/main/test-00000-of-00001.parquet",
            timeout=120,
        )
        resp.raise_for_status()
        with open(local_parquet, "wb") as f:
            f.write(resp.content)
        import pyarrow.parquet as pq
        tbl = pq.read_table(local_parquet)
        dataset = [{"question": r["question"], "answer": r["answer"]} for r in tbl.to_pylist()]
    total = len(dataset)
    print(flush=True, f"  Total questions: {total}\n")

    correct = 0
    errors = 0
    results = []
    start_time = time.time()

    for i, item in enumerate(dataset):
        if max_questions and i >= max_questions:
            break

        question = item["question"]
        true_answer_str = item["answer"]

        true_match = re.search(r'####\s*(-?\d+\.?\d*)', true_answer_str)
        if true_match:
            true_answer = true_match.group(1)
        else:
            errors += 1
            continue

        prompt = build_prompt(question)
        # Use the api_url from local scope
        output, finish_reason = call_model(prompt, api_url)
        if output is None:
            errors += 1
            continue

        pred_answer = extract_answer(output)
        is_correct = pred_answer == true_answer

        if is_correct:
            correct += 1

        results.append({
            "q_idx": i,
            "question": question[:80],
            "true_answer": true_answer,
            "pred_answer": pred_answer,
            "finish_reason": finish_reason,
            "output_snippet": output[:300],
            "correct": is_correct,
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            acc = correct / (i + 1 - errors) * 100
            print(flush=True, f"  [{i+1}/{total}] acc={acc:.2f}% ({elapsed:.0f}s)")

    elapsed = time.time() - start_time
    n_processed = min(max_questions or total, total, len(results))
    valid = n_processed - errors
    final_score = correct / valid * 100 if valid > 0 else 0

    print(flush=True, )
    print(flush=True, "=" * 60)
    print(flush=True, f"  Final GSM8K Score: {final_score:.2f}% ({correct}/{valid})")
    print(flush=True, f"  Processed: {n_processed}/{total} questions, Errors: {errors}")
    print(flush=True, f"  Time: {elapsed:.0f}s")
    print(flush=True, "=" * 60)

    report = {
        "model": MODEL_NAME,
        "api_url": API_URL,
        "few_shot": FEW_SHOT_COUNT,
        "score": final_score,
        "correct": correct,
        "n_processed": n_processed,
        "total_available": total,
        "errors": errors,
        "results": results,
    }
    with open("eval_gsm8k_base_correct_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(flush=True, f"\nResults saved to eval_gsm8k_base_correct_results.json")


if __name__ == "__main__":
    main()
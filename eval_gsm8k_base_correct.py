"""
GSM8K Corrected Evaluation for Qwen3-4B-Base
==============================================
Uses /v1/completions (not chat) with lm-eval standard 8-shot CoT prompt,
to match the official Qwen3 evaluation methodology.

Usage:
  python3 eval_gsm8k_base_correct.py

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
MAX_TOKENS = 512
TEMPERATURE = 0.0
TOP_P = 1.0
FEW_SHOT_COUNT = 8
# ── 8-shot GSM8K examples (lm-eval standard format) ────────────────────
FEW_SHOT_EXAMPLES = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Let's think step by step. There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted. The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Let's think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have total in total?
Let's think step by step. Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Let's think step by step. Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave 20 - 12 = 8 lollipops to Denny. The answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Let's think step by step. Shawn started with 5 toys. He got 2 from mom and 2 from dad, so he got 4 more. 5 + 4 = 9. The answer is 9.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Let's think step by step. There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Let's think step by step. Michael started with 58 golf balls. He lost 23 on Tuesday, so he had 58 - 23 = 35. Then he lost 2 more on Wednesday, so he had 35 - 2 = 33. The answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Let's think step by step. Olivia had 23 dollars. Each bagel cost 3 dollars, and she bought 5. So she spent 5 * 3 = 15 dollars. She has 23 - 15 = 8 dollars left. The answer is 8."""


def build_prompt(question: str) -> str:
    return f"{FEW_SHOT_EXAMPLES}\n\nQuestion: {question}\nLet's think step by step."


def extract_answer(text: str) -> str | None:
    match = re.search(r'The answer is\s*([+-]?\d+\.?\d*)', text, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r'\\?boxed\{([+-]?\d+\.?\d*)\}', text)
    if match:
        return match.group(1)
    match = re.search(r'ANSWER:\s*([+-]?\d+\.?\d*)', text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def call_model(prompt: str, url: str = API_URL) -> str | None:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stop": ["Question:"],
        "seed": 42,
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["text"]
    except Exception as e:
        print(f"  [ERROR] API call failed: {e}")
        return None


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

    print(f"GSM8K Evaluation (Corrected) — {MODEL_NAME}{scope_label}")
    print(f"  API: {api_url}")
    print(f"  Few-shot: {FEW_SHOT_COUNT}-shot (lm-eval standard)")
    print(f"  Format: /v1/completions (raw prompt, no chat template)\n")

    print("Loading GSM8K dataset...")
    local_parquet = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsm8k_test.parquet")
    try:
        import pyarrow.parquet as pq
        tbl = pq.read_table(local_parquet)
        dataset = [{"question": r["question"], "answer": r["answer"]} for r in tbl.to_pylist()]
    except Exception as e:
        print(f"  [WARN] Local parquet load failed: {e}")
        # Fallback: try to download with requests
        print("  Trying to download from HuggingFace Hub...")
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
    print(f"  Total questions: {total}\n")

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
        output = call_model(prompt, api_url)
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
            "output_snippet": output[:150],
            "correct": is_correct,
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            acc = correct / (i + 1 - errors) * 100
            print(f"  [{i+1}/{total}] acc={acc:.2f}% ({elapsed:.0f}s)")

    elapsed = time.time() - start_time
    valid = total - errors
    final_score = correct / valid * 100 if valid > 0 else 0

    print()
    print("=" * 60)
    print(f"  Final GSM8K Score: {final_score:.2f}% ({correct}/{valid})")
    print(f"  Errors: {errors}/{total}")
    print(f"  Time: {elapsed:.0f}s")
    print("=" * 60)

    report = {
        "model": MODEL_NAME,
        "api_url": API_URL,
        "few_shot": FEW_SHOT_COUNT,
        "score": final_score,
        "correct": correct,
        "total_valid": valid,
        "total": total,
        "errors": errors,
        "results": results,
    }
    with open("eval_gsm8k_base_correct_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to eval_gsm8k_base_correct_results.json")


if __name__ == "__main__":
    main()
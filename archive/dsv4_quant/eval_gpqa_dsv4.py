#!/usr/bin/env python3
"""
GPQA-Diamond evaluation for DeepSeek-V4-Flash-w8a8-mtp using evalscope.
Supports smoke test (--smoke N) and full run (198 questions).

Usage:
    python3 eval_gpqa_dsv4.py                    # Full 198 questions
    python3 eval_gpqa_dsv4.py --smoke 5          # Quick smoke test
    python3 eval_gpqa_dsv4.py --temperature 0    # Official eval temp
"""
import argparse
import json
import os
import sys
import time

API_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "dsv4"

def run_eval(smoke: int = 0, temperature: float = 0.0):
    """Run evalscope GPQA-Diamond evaluation."""

    print(f"[eval] Model: {MODEL_NAME}", flush=True)
    print(f"[eval] API:   {API_URL}", flush=True)
    print(f"[eval] Temp:  {temperature}", flush=True)
    print(f"[eval] Smoke: {'yes (' + str(smoke) + ' questions)' if smoke else 'no (full 198)'}", flush=True)
    print(flush=True)

    from evalscope import TaskConfig, run_task

    generation_config = {
        'temperature': temperature,
        'max_tokens': 8192,
        'n': 1,
    }
    if temperature == 0:
        generation_config['seed'] = 42

    task_kwargs = dict(
        model=MODEL_NAME,
        api_url=API_URL,
        eval_type='openai_api',
        datasets=['gpqa_diamond'],
        dataset_args={'gpqa_diamond': {}},
        eval_batch_size=8,
        generation_config=generation_config,
        timeout=300000,
        stream=True,
    )
    if smoke:
        task_kwargs['limit'] = smoke

    task_cfg = TaskConfig(**task_kwargs)

    print(f"[eval] TaskConfig prepared. Starting eval...", flush=True)
    sys.stdout.flush()
    start_ts = time.time()
    run_task(task_cfg=task_cfg)
    elapsed = time.time() - start_ts
    print(f"\n[eval] Done in {elapsed:.1f}s", flush=True)
# ── Parse results ──
    outputs_dir = "./outputs"
    if os.path.isdir(outputs_dir):
        dirs = sorted([d for d in os.listdir(outputs_dir)
                       if os.path.isdir(os.path.join(outputs_dir, d))])
        if dirs:
            latest = os.path.join(outputs_dir, dirs[-1])
            reviews_dir = os.path.join(latest, "reviews", MODEL_NAME)
            if os.path.isdir(reviews_dir):
                jsonl_files = [f for f in os.listdir(reviews_dir) if f.endswith(".jsonl")]
                if jsonl_files:
                    jsonl_path = os.path.join(reviews_dir, jsonl_files[0])
                    print(f"\n[eval] Review file: {jsonl_path}", flush=True)

                    correct = 0
                    total = 0
                    with open(jsonl_path) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                                score_info = rec.get("sample_score", {})
                                score_val = score_info.get("score", {})
                                acc = score_val.get("value", {}).get("acc")
                                if acc is not None:
                                    total += 1
                                    if acc == 1.0:
                                        correct += 1
                            except (json.JSONDecodeError, KeyError):
                                pass

                    if total > 0:
                        acc_pct = correct / total * 100
                        print(f"\n{'=' * 60}", flush=True)
                        print(f"  GPQA-Diamond Results ({MODEL_NAME})", flush=True)
                        print(f"  Accuracy: {correct}/{total} = {acc_pct:.2f}%", flush=True)
                        print(f"  Total time: {elapsed:.0f}s", flush=True)
                        print(f"  Official: 88.17 (1 Atlas 800 A3, w8a8-mtp)", flush=True)
                        print(f"{'=' * 60}", flush=True)

                        summary = {
                            "model": MODEL_NAME,
                            "dataset": "gpqa_diamond",
                            "accuracy": round(acc_pct, 2),
                            "correct": correct,
                            "total": total,
                            "temperature": temperature,
                            "time_s": round(elapsed, 0),
                            "official_score": 88.17,
                        }
                        with open("dsv4_gpqa_result.json", "w") as f:
                            json.dump(summary, f, indent=2)
                        print(f"\n  Results saved to dsv4_gpqa_result.json", flush=True)

            # Perf metrics from predictions
            preds_dir = os.path.join(latest, "predictions", MODEL_NAME)
            if os.path.isdir(preds_dir):
                pred_files = [f for f in os.listdir(preds_dir) if f.endswith(".jsonl")]
                if pred_files:
                    pred_path = os.path.join(preds_dir, pred_files[0])
                    latencies, ttfts, tpots = [], [], []
                    with open(pred_path) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                                for msg in rec.get("messages", []):
                                    perf = msg.get("perf_metrics")
                                    if perf:
                                        if perf.get("latency"):
                                            latencies.append(perf["latency"])
                                        if perf.get("ttft"):
                                            ttfts.append(perf["ttft"])
                                        if perf.get("tpot"):
                                            tpots.append(perf["tpot"])
                            except json.JSONDecodeError:
                                pass
                    if latencies:
                        avg_lat = sum(latencies) / len(latencies)
                        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0
                        avg_tpot = sum(tpots) / len(tpots) if tpots else 0
                        print(f"\n  Performance:", flush=True)
                        print(f"    Avg latency: {avg_lat:.2f}s", flush=True)
                        print(f"    Avg TTFT:    {avg_ttft * 1000:.1f}ms", flush=True)
                        print(f"    Avg TPOT:    {avg_tpot * 1000:.1f}ms", flush=True)
                        print(f"    Throughput:  {1 / avg_lat:.2f} req/s", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="GPQA-Diamond evaluation for DeepSeek-V4-Flash-w8a8-mtp")
    parser.add_argument('--smoke', type=int, default=0,
                        help='Run smoke test with N questions instead of full 198')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature (default 0.0 for official eval)')
    args = parser.parse_args()
    run_eval(smoke=args.smoke, temperature=args.temperature)


if __name__ == "__main__":
    main()
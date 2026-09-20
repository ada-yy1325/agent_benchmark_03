#!/usr/bin/env python3
"""
GPQA-Diamond evaluation for Qwen3-8B-Instruct using evalscope.
Supports FP16 (port 8811) and W8A8 (port 8812) modes.

Usage:
    python3 eval_gpqa_qwen3_8b.py --mode fp16      # FP16 baseline
    python3 eval_gpqa_qwen3_8b.py --mode w8a8      # W8A8 quantized
    python3 eval_gpqa_qwen3_8b.py --mode fp16 --smoke 5   # Smoke test
"""
import argparse
import json
import os
import sys
import time


def run_eval(mode: str = "fp16", smoke: int = 0):
    """Run evalscope GPQA-Diamond evaluation."""

    if mode == "fp16":
        port = 8811
        model_name = "Qwen3-8B-FP16"
    elif mode == "w8a8":
        port = 8812
        model_name = "Qwen3-8B-W8A8"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    api_url = f"http://127.0.0.1:{port}/v1/chat/completions"
    print(f"[eval] Mode: {mode.upper()}", flush=True)
    print(f"[eval] Model: {model_name}", flush=True)
    print(f"[eval] API:   {api_url}", flush=True)
    print(f"[eval] Smoke: {'yes (' + str(smoke) + ' questions)' if smoke else 'no (full 198)'}", flush=True)
    print(flush=True)

    from evalscope import TaskConfig, run_task

    dataset_args = {
        'gpqa_diamond': {
            'few_shot_num': 0,
        }
    }

    generation_config = {
        'temperature': 0,
        'seed': 42,
        'top_p': 1.0,
        'top_k': -1,
        # IMPORTANT: max_tokens must leave headroom for the input prompt below the
        # server's --max-model-len, otherwise vLLM rejects the request
        # ("requested N output tokens ... upper bound for 0 input tokens").
        # FP16 server: max-model-len 32768; W8A8 server: max-model-len 8192.
        # 4096 leaves >=4096 input tokens on W8A8, ample for GPQA few_shot=0 prompts.
        'max_tokens': 4096,
        'n': 1,
    }

    task_kwargs = dict(
        model=model_name,
        api_url=api_url,
        eval_type='openai_api',
        datasets=['gpqa_diamond'],
        dataset_args=dataset_args,
        eval_batch_size=16,
        generation_config=generation_config,
        timeout=120000,
        stream=True,
    )
    if smoke:
        task_kwargs['limit'] = smoke  # limit at TaskConfig level
    task_cfg = TaskConfig(**task_kwargs)

    print(f"[eval] TaskConfig prepared. Starting eval...", flush=True)
    sys.stdout.flush()
    start_ts = time.time()
    run_task(task_cfg=task_cfg)
    elapsed = time.time() - start_ts
    print(f"\n[eval] Done in {elapsed:.1f}s", flush=True)

    parse_results(model_name, elapsed)


def parse_results(model_name: str, elapsed: float):
    """Parse latest evalscope output for accuracy and perf metrics."""
    outputs_dir = "./outputs"
    if not os.path.isdir(outputs_dir):
        return

    dirs = sorted([d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))])
    if not dirs:
        return

    latest = os.path.join(outputs_dir, dirs[-1])
    reviews_dir = os.path.join(latest, "reviews", model_name)
    if not os.path.isdir(reviews_dir):
        return

    jsonl_files = [f for f in os.listdir(reviews_dir) if f.endswith(".jsonl")]
    if not jsonl_files:
        return

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
        print(f"\n{'=' * 60}", flush=True)
        print(f"  GPQA-Diamond Results ({model_name})", flush=True)
        print(f"  Accuracy: {correct}/{total} = {correct/total*100:.2f}%", flush=True)
        print(f"  Total time: {elapsed:.0f}s", flush=True)
        print(f"{'=' * 60}", flush=True)

    preds_dir = os.path.join(latest, "predictions", model_name)
    if os.path.isdir(preds_dir):
        pred_files = [f for f in os.listdir(preds_dir) if f.endswith(".jsonl")]
        if pred_files:
            pred_path = os.path.join(preds_dir, pred_files[0])
            latencies = []
            ttfts = []
            tpots = []
            with open(pred_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        msgs = rec.get("messages", [])
                        for msg in msgs:
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
                print(f"\n  Performance ({model_name}):", flush=True)
                print(f"    Avg latency: {avg_lat:.2f}s", flush=True)
                print(f"    Avg TTFT:    {avg_ttft*1000:.1f}ms", flush=True)
                print(f"    Avg TPOT:    {avg_tpot*1000:.1f}ms", flush=True)
                print(f"    Throughput:  {1/avg_lat:.2f} req/s", flush=True)


def main():
    parser = argparse.ArgumentParser(description="GPQA-Diamond evaluation for Qwen3-8B-Instruct")
    parser.add_argument('--mode', choices=['fp16', 'w8a8'], required=True,
                        help='Evaluation mode: fp16 or w8a8')
    parser.add_argument('--smoke', type=int, default=0,
                        help='Run smoke test with N questions instead of full 198')
    args = parser.parse_args()
    run_eval(mode=args.mode, smoke=args.smoke)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""GPQA runner for w8a8-mtp model with CORRECT result parsing (sample_score schema).

Usage: python3 /tmp/gpqa_mtp.py [smoke_count] [out_dir]
"""
import json, os, sys, time

API_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "dsv4"
smoke = int(sys.argv[1]) if len(sys.argv) > 1 else 0
out_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/phase5/mtp_model"

from evalscope import TaskConfig, run_task

gen_cfg = {'temperature': 0.0, 'max_tokens': 8192, 'n': 1, 'seed': 42}
task_kwargs = dict(
    model=MODEL, api_url=API_URL, eval_type='openai_api',
    datasets=['gpqa_diamond'], dataset_args={'gpqa_diamond': {}},
    eval_batch_size=8, generation_config=gen_cfg, timeout=300000, stream=True)
if smoke:
    task_kwargs['limit'] = smoke

print(f"[gpqa-mtp] starting {'smoke(%d)' % smoke if smoke else 'FULL 198'}", flush=True)
t0 = time.time()
run_task(task_cfg=TaskConfig(**task_kwargs))
dt = time.time() - t0
print(f"[gpqa-mtp] eval done in {dt:.1f}s", flush=True)

outputs_dir = "./outputs"
dirs = sorted([d for d in os.listdir(outputs_dir)
               if os.path.isdir(os.path.join(outputs_dir, d)) and d.startswith("2026")])
latest = os.path.join(outputs_dir, dirs[-1])
reviews_dir = os.path.join(latest, "reviews", MODEL)
files = [f for f in os.listdir(reviews_dir) if f.endswith(".jsonl")] if os.path.isdir(reviews_dir) else []

if files:
    jpath = os.path.join(reviews_dir, files[0])
    correct = total = 0
    sample_pred = None
    for line in open(jpath):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        total += 1
        sc = d.get("sample_score", {})
        acc = sc.get("score", {}).get("value", {}).get("acc", 0.0)
        if acc and acc > 0:
            correct += 1
        if sample_pred is None:
            sample_pred = sc.get("score", {}).get("value", {}).get("prediction", "")
    res = {"model": MODEL, "dataset": "gpqa_diamond", "num_questions": total,
           "correct": correct, "accuracy_pct": round(correct / total * 100, 2),
           "elapsed_s": round(dt, 1), "eval_dir": latest,
           "sample_prediction": str(sample_pred)[:120]}
    with open(f"{out_dir}/gpqa_{'smoke' if smoke else 'full'}_result.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"[gpqa-mtp] ACCURACY: {correct}/{total} = {correct/total*100:.2f}%", flush=True)
else:
    print("[gpqa-mtp] no reviews found; latest:", latest, flush=True)

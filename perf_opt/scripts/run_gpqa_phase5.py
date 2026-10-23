#!/usr/bin/env python3
"""Phase 5.5: GPQA-Diamond evaluation on final config (static_kernel=ON, fp8 KV cache)."""
import subprocess, sys, os, argparse

API_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "dsv4"
REPO_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test"
OUTPUT_LOG = "/tmp/phase5/gpqa_phase5.log"

def run_gpqa(smoke: int = 0):
    """Run GPQA via evalscope directly in the working dir."""
    script = f"""
import json, os, sys, time
API_URL = "{API_URL}"
MODEL = "{MODEL}"
SMOKE = {smoke}

print(f"[gpqa] Model: {{MODEL}}", flush=True)
print(f"[gpqa] API: {{API_URL}}", flush=True)
print(f"[gpqa] Smoke: {{'yes (' + str(SMOKE) + ' questions)' if SMOKE else 'no (full 198)'}}", flush=True)

from evalscope import TaskConfig, run_task

gen_cfg = {{'temperature': 0.0, 'max_tokens': 8192, 'n': 1, 'seed': 42}}
task_kwargs = dict(
    model=MODEL, api_url=API_URL, eval_type='openai_api',
    datasets=['gpqa_diamond'], dataset_args={{'gpqa_diamond': {{}}}},
    eval_batch_size=8, generation_config=gen_cfg, timeout=300000, stream=True,
)
if SMOKE:
    task_kwargs['limit'] = SMOKE

task_cfg = TaskConfig(**task_kwargs)
print(f"[gpqa] Starting eval...", flush=True)
start = time.time()
run_task(task_cfg=task_cfg)
elapsed = time.time() - start
print(f"[gpqa] Done in {{elapsed:.1f}}s", flush=True)

# Parse results
outputs_dir = "./outputs"
dirs = sorted([d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))])
if dirs:
    latest = os.path.join(outputs_dir, dirs[-1])
    reviews_dir = os.path.join(latest, "reviews", MODEL)
    if os.path.isdir(reviews_dir):
        jsonl_files = [f for f in os.listdir(reviews_dir) if f.endswith(".jsonl")]
        if jsonl_files:
            jpath = os.path.join(reviews_dir, jsonl_files[0])
            correct = 0; total = 0
            with open(jpath) as f:
                for line in f:
                    if not line.strip(): continue
                    d = json.loads(line)
                    total += 1
                    if d.get("score", False): correct += 1
            print(f"[gpqa] ACCURACY: {{correct}}/{{total}} = {{correct/total*100:.2f}}%", flush=True)
            # Save result JSON
            res = {{"model": MODEL, "dataset": "gpqa_diamond", "num_questions": total,
                    "correct": correct, "accuracy_pct": round(correct/total*100, 2),
                    "elapsed_s": round(elapsed, 1)}}
            with open("/tmp/phase5/gpqa_phase5_result.json", "w") as f:
                json.dump(res, f, indent=2)
            print(f"[gpqa] Result saved to /tmp/phase5/gpqa_phase5_result.json", flush=True)
    else:
        print("[gpqa] ⚠️ No reviews dir found", flush=True)
else:
    print("[gpqa] ⚠️ No outputs dir found", flush=True)
"""
    cmd = [sys.executable, "-c", script]
    print(f"[runner] GPQA starting {'(smoke=' + str(smoke) + ')' if smoke else '(FULL 198)'}")
    with open(OUTPUT_LOG, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO_DIR)
    print(f"[runner] Exit code: {result.returncode}")
    # Print last lines
    with open(OUTPUT_LOG) as f:
        lines = f.readlines()
        for line in lines[-20:]:
            print(line.rstrip())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=int, default=0)
    args = parser.parse_args()
    run_gpqa(args.smoke)
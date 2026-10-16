"""
MMLU 评测脚本（lm-eval-harness + vLLM 服务）
对标 Red Hat 口径：
  - 0-shot, raw text prompting, log-likelihood scoring
  - 使用 lm_eval --model openai-completions 对接已启动的 vLLM 服务

用法：
  python3 run_mmlu_eval.py fp16      # FP16 基线（port 8811）
  python3 run_mmlu_eval.py w8a8      # Red Hat W8A8（port 8812）
"""

import sys, os, json, time, subprocess, urllib.request

ROOT = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test"

SERVERS = {
    "fp16": {"model_name": "Llama-3.1-8B-Instruct-FP16", "port": 8811},
    "w8a8": {"model_name": "Llama-3.1-8B-Instruct-W8A8", "port": 8812},
}
OUTPUT_NAMES = {"fp16": "mmlu_fp16", "w8a8": "mmlu_w8a8"}
LOG_FILES = {"fp16": "mmlu_fp16.log", "w8a8": "mmlu_w8a8.log"}


def check_server(config):
    url = f"http://localhost:{config['port']}/v1/completions"
    data = json.dumps({
        "model": config["model_name"],
        "prompt": "The capital of France is",
        "max_tokens": 1, "logprobs": 1, "echo": True,
    }).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            print(f"[OK] 服务在线: {result.get('model', 'unknown')} @ port {config['port']}")
            return True
    except Exception as e:
        print(f"[ERROR] 服务不可用 (port {config['port']}): {e}")
        return False


def run_mmlu(config_key: str):
    c = SERVERS[config_key]
    output_name = OUTPUT_NAMES[config_key]
    log_file = LOG_FILES[config_key]

    if not check_server(c):
        sys.exit(1)

    cmd = (
        f"cd {ROOT} && lm_eval --model openai-completions "
        f"--model_args "
        f"model={c['model_name']},"
        f"base_url=http://localhost:{c['port']}/v1/completions,"
        f"num_concurrent=1,"
        f"max_retries=3,"
        f"tokenized_requests=False,"
        f"tokenizer_backend=huggingface,"
        f"tokenizer_path={ROOT}/models/Meta-Llama-3.1-8B-Instruct "
        f"--tasks mmlu --num_fewshot 0 "
        f"--output_path ./outputs/{output_name} "
        f"--seed 42 "
        f"2>&1 | tee {log_file}"
    )

    print("=" * 60)
    print(f"MMLU 评测: {config_key.upper()}")
    print(f"模型: {c['model_name']} @ port {c['port']}")
    print(f"日志: {log_file}")
    print("=" * 60)

    session = f"mmlu_{config_key}"
    subprocess.run(
        f"tmux kill-session -t {session} 2>/dev/null; "
        f"tmux new-session -d -s {session} \"{cmd}\"",
        shell=True,
    )
    print(f"tmux: {session} | 查看: tmux capture-pane -t {session} -p -S -20\n")

    start = time.time()
    last = ""
    while True:
        time.sleep(30)
        ret = subprocess.run(
            f"tmux has-session -t {session}",
            shell=True, capture_output=True,
        )
        elapsed = time.time() - start
        if ret.returncode != 0:
            print(f"\n[完成] 耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
            break
        try:
            with open(f"{ROOT}/{log_file}") as f:
                for line in f.readlines()[-5:]:
                    line = line.strip()
                    if line and line != last and any(
                        k in line.lower() for k in ["acc", "mmlu", "running", "%"]
                    ):
                        print(f"  [{elapsed:.0f}s] {line[:150]}")
                        last = line
        except:
            pass
        if elapsed > 7200:
            print("[WARNING] 超 2h，终止")
            subprocess.run(f"tmux send-keys -t {session} C-c", shell=True)
            break

    print("\n" + "=" * 60)
    print(f"结果 ({config_key.upper()})")
    print("=" * 60)
    try:
        with open(f"{ROOT}/{log_file}") as f:
            for line in f:
                if "mmlu" in line.lower() and "acc" in line.lower():
                    print(f"  {line.rstrip()}")
    except:
        pass
    print(f"\n日志: {ROOT}/{log_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in SERVERS:
        print("用法: python3 run_mmlu_eval.py <fp16|w8a8>")
        sys.exit(1)
    run_mmlu(sys.argv[1])
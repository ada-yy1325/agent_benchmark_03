#!/usr/bin/env python3
"""
FP16 smoke test for GPQA-Diamond Qwen3-8B-Instruct.
Starts vLLM server, runs --smoke 5, stops server.
"""
import subprocess, time, sys, os, signal, urllib.request

BASEDIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test"
os.chdir(BASEDIR)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Starting FP16 vLLM server...")
proc = subprocess.Popen(
    [sys.executable, "start_vllm_8b_fp16.py"],
    stdout=open("vllm_fp16_gpqa.log", "w"),
    stderr=subprocess.STDOUT,
)
log(f"vLLM PID: {proc.pid}")

log("Waiting for server (up to 180s)...")
ready = False
for i in range(90):
    time.sleep(2)
    try:
        urllib.request.urlopen("http://127.0.0.1:8811/v1/models", timeout=3)
        log(f"Server ready after {(i+1)*2}s")
        ready = True
        break
    except Exception:
        pass

if not ready:
    log("FAIL: Server failed to start")
    with open("vllm_fp16_gpqa.log") as f:
        print(f.read()[-3000:])
    proc.kill()
    sys.exit(1)

log("Running FP16 eval (smoke=5)...")
eval_result = subprocess.run(
    [sys.executable, "eval_gpqa_qwen3_8b.py", "--mode", "fp16", "--smoke", "5"],
    capture_output=True, text=True,
)
print(eval_result.stdout)
if eval_result.stderr:
    print("STDERR:", eval_result.stderr[-1500:])

log("Stopping server...")
proc.terminate()
try:
    proc.wait(timeout=10)
except subprocess.TimeoutExpired:
    proc.kill()
log("Done!")
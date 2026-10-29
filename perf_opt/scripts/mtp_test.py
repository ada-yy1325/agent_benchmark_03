#!/usr/bin/env python3
"""Minimal test: start decode with MTP enabled, capture error immediately."""
import os, sys, subprocess, time, json

MODEL_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-npu16"

# Build args for decode instance WITH MTP
args = [
    MODEL_DIR, "--host", "0.0.0.0", "--port", "8002",
    "--max-model-len", "133120", "--max-num-batched-tokens", "8192",
    "--served-model-name", "dsv4_mtp_test", "--gpu-memory-utilization", "0.9",
    "--max-num-seqs", "8", "--data-parallel-size", "1",
    "--tensor-parallel-size", "8", "--enable-expert-parallel",
    "--quantization", "ascend", "--block-size", "128",
    "--compilation-config",
    '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [8, 16, 24, 32, 40, 48, 56, 64]}',
    "--speculative-config",
    '{"num_speculative_tokens": 1, "method": "mtp"}',
    "--additional-config",
    '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"enable_dsa_cp":true,"multistream_overlap_shared_expert":true}',
    "--tokenizer-mode", "deepseek_v4",
    "--tool-call-parser", "deepseek_v4",
    "--enable-auto-tool-choice",
    "--reasoning-parser", "deepseek_v4",
]

# Write args and launch
logfile = "/tmp/pd_test/mtp_test.log"
if os.path.exists(logfile):
    os.remove(logfile)

os.makedirs("/tmp/pd_test", exist_ok=True)
args_file = "/tmp/pd_test/mtp_test_args.json"
with open(args_file, "w") as f:
    json.dump(args, f)

script = f"""#!/bin/bash
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024 TASK_QUEUE_ENABLE=1 HCCL_OP_EXPANSION_MODE=AIV VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
cd /inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test
exec python3 -B -c "
import json, os
args = json.load(open('{args_file}'))
from vllm.entrypoints.openai.api_server import main
import sys; sys.argv = ['vllm_serve'] + args
main()
" 2>&1 | tee {logfile}
"""

script_path = "/tmp/pd_test/_mtp_test_launch.sh"
with open(script_path, "w") as f:
    f.write(script)
os.chmod(script_path, 0o755)

os.system(f"tmux kill-session -t mtp_test 2>/dev/null")
os.system(f"tmux new-session -d -s mtp_test bash {script_path}")
print("[launcher] Started tmux mtp_test")

# Wait up to 120s and capture error
for i in range(1, 121):
    time.sleep(1)
    if os.path.exists(logfile):
        with open(logfile) as f:
            content = f.read()
        if "Error" in content or "Traceback" in content or "assert" in content:
            print(f"\n[MTP test] Error found at {i}s:")
            print(content[-3000:])
            break
        if "Uvicorn running" in content:
            print(f"\n[MTP test] Server started successfully at {i}s!")
            break
    if i % 10 == 0:
        print(f"  ... waiting {i}s")
else:
    print("[MTP test] Timeout (120s) — checking log:")
    if os.path.exists(logfile):
        with open(logfile) as f:
            content = f.read()
        print(content[-2000:])
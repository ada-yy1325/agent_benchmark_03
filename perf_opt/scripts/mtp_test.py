#!/usr/bin/env python3
"""Quick MTP error reproduction — decode instance only."""
import json, sys, os, subprocess, traceback

cc = json.dumps({"cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [8,16,24,32,40,48,56,64]})
spec = json.dumps({"num_speculative_tokens": 1, "method": "mtp"})
adc = json.dumps({"ascend_compilation_config":{"enable_npugraph_ex":True,"enable_static_kernel":False},
    "enable_cpu_binding":True,"enable_dsa_cp":True,"multistream_overlap_shared_expert":True})

cmd = [
    "vllm", "serve",
    "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-npu16",
    "--host", "0.0.0.0", "--port", "8002",
    "--max-model-len", "133120", "--max-num-batched-tokens", "8192",
    "--served-model-name", "dsv4_mtp_test", "--gpu-memory-utilization", "0.9",
    "--max-num-seqs", "8",
    "--tensor-parallel-size", "8", "--enable-expert-parallel",
    "--quantization", "ascend", "--block-size", "128",
    "--compilation-config", cc,
    "--speculative-config", spec,
    "--additional-config", adc,
    "--tokenizer-mode", "deepseek_v4",
    "--tool-call-parser", "deepseek_v4",
    "--enable-auto-tool-choice",
    "--reasoning-parser", "deepseek_v4",
]

env = os.environ.copy()
env.update({
    "OMP_PROC_BIND": "false", "OMP_NUM_THREADS": "10",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_BUFFSIZE": "1024", "TASK_QUEUE_ENABLE": "1",
    "HCCL_OP_EXPANSION_MODE": "AIV", "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
    "ASCEND_RT_VISIBLE_DEVICES": "8,9,10,11,12,13,14,15",
})

print(f"[MTP test] Running: {' '.join(cmd)}", flush=True)
try:
    result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=90, text=True)
    print(result.stdout[-3000:])
except subprocess.TimeoutExpired as e:
    print(f"[MTP test] Timeout after 90s. Output so far:\n{e.stdout[-3000:] if e.stdout else '(none)'}")
except Exception as e:
    traceback.print_exc()
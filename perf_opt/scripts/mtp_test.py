#!/usr/bin/env python3
"""Quick MTP error reproduction — decode instance only."""
import json, sys, os, traceback

os.environ.setdefault('OMP_PROC_BIND', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '10')
os.environ.setdefault('PYTORCH_NPU_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('HCCL_BUFFSIZE', '1024')
os.environ.setdefault('TASK_QUEUE_ENABLE', '1')
os.environ.setdefault('HCCL_OP_EXPANSION_MODE', 'AIV')
os.environ.setdefault('VLLM_ASCEND_ENABLE_FLASHCOMM1', '1')
os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '8,9,10,11,12,13,14,15')

CC = json.dumps({"cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [8,16,24,32,40,48,56,64]})
SPEC = json.dumps({"num_speculative_tokens": 1, "method": "mtp"})
ADC = json.dumps({"ascend_compilation_config":{"enable_npugraph_ex":True,"enable_static_kernel":False},
    "enable_cpu_binding":True,"enable_dsa_cp":True,"multistream_overlap_shared_expert":True})

args = [
    "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-npu16",
    "--host", "0.0.0.0", "--port", "8002",
    "--max-model-len", "133120", "--max-num-batched-tokens", "8192",
    "--served-model-name", "dsv4_mtp_test", "--gpu-memory-utilization", "0.9",
    "--max-num-seqs", "8",
    "--tensor-parallel-size", "8", "--enable-expert-parallel",
    "--quantization", "ascend", "--block-size", "128",
    "--compilation-config", CC,
    "--speculative-config", SPEC,
    "--additional-config", ADC,
    "--tokenizer-mode", "deepseek_v4",
    "--tool-call-parser", "deepseek_v4",
    "--enable-auto-tool-choice",
    "--reasoning-parser", "deepseek_v4",
]

sys.argv = ["vllm_serve"] + args
try:
    from vllm.entrypoints.openai.api_server import main
    main()
except Exception:
    traceback.print_exc()
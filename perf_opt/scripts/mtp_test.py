#!/usr/bin/env python3
"""MTP test - start vLLM serve with MTP enabled on cards 8-15"""
# MUST be at the very top, before any other imports
import torch._functorch.config as _fc
_fc.enable_autograd_cache = False
_fc.bypass_autograd_cache_key = True

import json, sys, os

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "8,9,10,11,12,13,14,15")
os.environ.setdefault("OMP_PROC_BIND", "false")
os.environ.setdefault("OMP_NUM_THREADS", "10")

MODEL = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-npu16"

cc = json.dumps({"cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [8, 16, 24, 32, 40, 48, 56, 64]})
spec = json.dumps({"num_speculative_tokens": 1, "method": "mtp"})
adc = json.dumps({"ascend_compilation_config": {"enable_npugraph_ex": True,
    "enable_static_kernel": False}, "enable_cpu_binding": True,
    "enable_dsa_cp": True, "multistream_overlap_shared_expert": True})

sys.argv = ["vllm", "serve", MODEL,
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

from vllm.entrypoints.openai.api_server import run_server
run_server()
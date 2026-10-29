#!/usr/bin/env python3
"""MTP test - start vLLM serve with MTP enabled on cards 8-15"""
# MUST be at the very top, before any other imports
import torch._functorch.config as _fc
_fc.enable_autograd_cache = False
_fc.bypass_autograd_cache_key = True

import os, sys

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "8,9,10,11,12,13,14,15")
os.environ.setdefault("OMP_PROC_BIND", "false")
os.environ.setdefault("OMP_NUM_THREADS", "10")

sys.argv = [
    "vllm", "serve",
    "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-npu16",
    "--host", "0.0.0.0", "--port", "8002",
    "--max-model-len", "133120", "--max-num-batched-tokens", "8192",
    "--served-model-name", "dsv4_mtp_test", "--gpu-memory-utilization", "0.9",
    "--max-num-seqs", "64",
    "--tensor-parallel-size", "8", "--enable-expert-parallel",
    "--quantization", "ascend", "--block-size", "128",
    "--model-loader-extra-config", '{"enable_multithread_load":true,"num_threads":128}',
    # graph mode: cudagraph FULL_DECODE_ONLY (same as start_server_phase5 default)
    "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
    # MTP spec: enforce_eager INSIDE speculative-config (production pattern)
    "--speculative-config", '{"num_speculative_tokens":1,"method":"mtp","enforce_eager":true}',
    # additional: dsa_cp ON (production), npugraph_ex + static_kernel ON
    "--additional-config", '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":true},"enable_cpu_binding":true,"enable_dsa_cp":true,"multistream_overlap_shared_expert":true}',
    "--tokenizer-mode", "deepseek_v4",
    "--tool-call-parser", "deepseek_v4",
    "--enable-auto-tool-choice",
    "--reasoning-parser", "deepseek_v4",
]

from vllm.entrypoints.cli.main import main

if __name__ == "__main__":
    main()
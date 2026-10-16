#!/usr/bin/env python3
"""Quick test: can TP=4 fit in 64 GB HBM?"""
import os, sys, asyncio, argparse, time

os.environ.setdefault("OMP_PROC_BIND", "false")
os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HCCL_BUFFSIZE", "1024")
os.environ.setdefault("TASK_QUEUE_ENABLE", "1")
os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "AIV")
os.environ.setdefault("VLLM_ASCEND_ENABLE_FLASHCOMM1", "1")

import torch._functorch.config as _fc
_fc.enable_autograd_cache = False
_fc.bypass_autograd_cache_key = True

TP = 4
PORT = 8000
BATCHED = 8192
MAX_SEQS = 64

MODEL_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-self"

CLI_ARGS = [
    MODEL_DIR,
    "--host", "0.0.0.0",
    "--port", str(PORT),
    "--max-model-len", "133120",
    "--max-num-batched-tokens", str(BATCHED),
    "--served-model-name", "dsv4",
    "--gpu-memory-utilization", "0.9",
    "--max-num-seqs", str(MAX_SEQS),
    "--data-parallel-size", "1",
    "--tensor-parallel-size", str(TP),
    "--enable-expert-parallel",
    "--tokenizer-mode", "deepseek_v4",
    "--tool-call-parser", "deepseek_v4",
    "--enable-auto-tool-choice",
    "--reasoning-parser", "deepseek_v4",
    "--no-enable-prefix-caching",
    "--model-loader-extra-config", '{"enable_multithread_load": true, "num_threads": 128}',
    "--quantization", "ascend",
    "--block-size", "128",
    "--speculative-config", '{"num_speculative_tokens": 1, "method": "mtp", "enforce_eager": true}',
    "--compilation-config", '{"cudagraph_mode": "FULL_DECODE_ONLY"}',
    "--additional-config", '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding": true,"enable_dsa_cp": true,"multistream_overlap_shared_expert": true}',
]

if __name__ == "__main__":
    print(f"[launcher] TP={TP} port={PORT}", flush=True)
    devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "NOT SET")
    print(f"[launcher] ASCEND_RT_VISIBLE_DEVICES={devices}", flush=True)

    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args

    p = FlexibleArgumentParser(description="vLLM API server (TP4 test)")
    p = make_arg_parser(p)
    a = p.parse_args(CLI_ARGS)
    validate_parsed_serve_args(a)

    if hasattr(a, "model_tag") and a.model_tag is not None:
        a.model = a.model_tag
    if hasattr(a, "tokenizer_tag") and a.tokenizer_tag is not None:
        a.tokenizer = a.tokenizer_tag

    from vllm.entrypoints.openai.api_server import run_server
    print(f"[launcher] Starting TP4 server...", flush=True)
    t0 = time.time()
    asyncio.run(run_server(a))
    print(f"[launcher] TP4 server started in {time.time()-t0:.0f}s", flush=True)
#!/usr/bin/env python3
"""
Baseline vLLM server launcher for DeepSeek-V4-Flash-w8a8-self.
All parameters are identical to the verified self-quant evaluation config (TP=8, MTP on).
DO NOT change any parameter — this is the frozen baseline.
"""
import os
import sys
import asyncio

# ── Step 1: Disable AOTAutogradCache BEFORE any vLLM import ──────────
import torch._functorch.config as _functorch_cfg
_functorch_cfg.enable_autograd_cache = False
_functorch_cfg.bypass_autograd_cache_key = True
print("[launcher] ✓ AOTAutogradCache disabled", flush=True)

# ── Step 2: Set environment variables (official A3 config) ──
os.environ.setdefault("OMP_PROC_BIND", "false")
os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HCCL_BUFFSIZE", "1024")
os.environ.setdefault("TASK_QUEUE_ENABLE", "1")
os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "AIV")
os.environ.setdefault("VLLM_ASCEND_ENABLE_FLASHCOMM1", "1")

# ── MODEL_DIR: verified by step 0.1, default points to self-quant model ──
MODEL_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-self"

# ── CLI args (frozen baseline — do not modify) ──
CLI_ARGS = [
    MODEL_DIR,
    '--host', '0.0.0.0',
    '--port', '8000',
    '--max-model-len', '133120',
    '--max-num-batched-tokens', '8192',
    '--served-model-name', 'dsv4',
    '--gpu-memory-utilization', '0.9',
    '--max-num-seqs', '32',
    '--data-parallel-size', '1',
    '--tensor-parallel-size', '8',
    '--enable-expert-parallel',
    '--tokenizer-mode', 'deepseek_v4',
    '--tool-call-parser', 'deepseek_v4',
    '--enable-auto-tool-choice',
    '--reasoning-parser', 'deepseek_v4',
    '--no-enable-prefix-caching',
    '--model-loader-extra-config', '{"enable_multithread_load": true, "num_threads": 128}',
    '--quantization', 'ascend',
    '--block-size', '128',
    '--speculative-config', '{"num_speculative_tokens": 1, "method": "mtp", "enforce_eager": true}',
    '--compilation-config', '{"cudagraph_mode": "FULL_DECODE_ONLY"}',
    '--additional-config', '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding": true,"enable_dsa_cp": true,"multistream_overlap_shared_expert": true}',
]

if __name__ == "__main__":
    print(f"[launcher] MODEL_DIR: {MODEL_DIR}", flush=True)
    print(f"[launcher] Starting vLLM (baseline config, TP=8, MTP=on)", flush=True)
    print(f"  {' '.join(CLI_ARGS)}", flush=True)
    sys.stdout.flush()

    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args

    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args(CLI_ARGS)
    validate_parsed_serve_args(args)

    if hasattr(args, 'model_tag') and args.model_tag is not None:
        args.model = args.model_tag
    if hasattr(args, 'tokenizer_tag') and args.tokenizer_tag is not None:
        args.tokenizer = args.tokenizer_tag

    print(f"[launcher] Starting server... model={args.model}, port={args.port}", flush=True)
    from vllm.entrypoints.openai.api_server import run_server
    asyncio.run(run_server(args))
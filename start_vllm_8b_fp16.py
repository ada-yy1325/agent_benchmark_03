#!/usr/bin/env python3
"""
vLLM server launcher for Qwen3-8B-Instruct FP16 (GPQA-Diamond test).
Port 8811, served-model-name Qwen3-8B-FP16.
"""
import sys
import os
import asyncio

# ── Step 1: Disable AOTAutogradCache BEFORE any vLLM import ──────────
import torch._functorch.config as _functorch_cfg
_functorch_cfg.enable_autograd_cache = False
_functorch_cfg.bypass_autograd_cache_key = True
print("[patcher] ✓ AOTAutogradCache disabled", flush=True)

# ── Step 2: Define MODEL_DIR ──
# NOTE: /inspire/ shared filesystem does NOT support symlink resolution.
# Use the real path directly instead of models/Qwen3-8B symlink.
MODEL_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/Qwen/Qwen3-8B"

# ── Step 3: Build CLI args ──
CLI_ARGS = [
    MODEL_DIR,
    '--port', '8811',
    '--max-model-len', '32768',
    '--dtype', 'auto',
    '--served-model-name', 'Qwen3-8B-FP16',
    '--trust-remote-code',
    '--gpu-memory-utilization', '0.9',
    '--enforce-eager',
]

# ── Step 4: Parse args and start vLLM ──
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args

parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
parser = make_arg_parser(parser)
args = parser.parse_args(CLI_ARGS)
validate_parsed_serve_args(args)

# ── Fix: model_tag → model ──
if hasattr(args, 'model_tag') and args.model_tag is not None:
    args.model = args.model_tag
if hasattr(args, 'tokenizer_tag') and args.tokenizer_tag is not None:
    args.tokenizer = args.tokenizer_tag

print(f"[patcher] Starting vLLM with: {' '.join(CLI_ARGS)}", flush=True)
print(f"[patcher]   model={args.model}, served_model_name={args.served_model_name}", flush=True)

from vllm.entrypoints.openai.api_server import run_server
asyncio.run(run_server(args))
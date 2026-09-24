#!/usr/bin/env python3
"""
vLLM server launcher for Red Hat Meta-Llama-3.1-8B-Instruct W8A8 (GPQA-Diamond test).
Port 8812, served-model-name Llama-3.1-8B-Instruct-W8A8.
"""
import sys
import os
import asyncio
import json

# ── Step 1: Disable AOTAutogradCache BEFORE any vLLM import ──────────
import torch._functorch.config as _functorch_cfg
_functorch_cfg.enable_autograd_cache = False
_functorch_cfg.bypass_autograd_cache_key = True
print("[patcher] ✓ AOTAutogradCache disabled", flush=True)

# ── Step 2: Define MODEL_DIR ──
MODEL_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/Meta-Llama-3.1-8B-Instruct-W8A8-RedHat"

# ── Step 3: Build CLI args ──
CLI_ARGS = [
    MODEL_DIR,
    '--port', '8812',
    '--max-model-len', '32768',
    '--dtype', 'auto',
    '--served-model-name', 'Llama-3.1-8B-Instruct-W8A8',
    '--trust-remote-code',
    '--gpu-memory-utilization', '0.9',
    '--enforce-eager',
    '--quantization', 'ascend',
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
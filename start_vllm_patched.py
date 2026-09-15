#!/usr/bin/env python3
"""
Patched vLLM server launcher.
Disables AOTAutogradCache before vLLM loads, to work around
the AssertionError: expected OutputCode, got _CompiledFxGraph
bug in PyTorch 2.11 + vLLM-Ascend integration.
"""
import sys
import os

# ── Step 1: Disable AOTAutogradCache BEFORE any vLLM import ──────────
import torch._functorch.config as _functorch_cfg
_functorch_cfg._config['enable_autograd_cache'] = False
# Also disable related cache paths that might cause similar issues
_functorch_cfg._config['bypass_autograd_cache_key'] = True
print("[patcher] ✓ AOTAutogradCache disabled", flush=True)

# ── Step 2: Build CLI args ──
MODEL_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/Qwen3-4B-W8A8"
CLI_ARGS = [
    MODEL_DIR,
    '--port', '8801',
    '--max-model-len', '8192',
    '--quantization', 'ascend',
    '--dtype', 'auto',
    '--served-model-name', 'Qwen3-4B',
]

# ── Step 3: Parse args and start vLLM ──
from vllm.entrypoints.openai.cli import make_arg_parser
parser = make_arg_parser()
args = parser.parse_args(CLI_ARGS)
print(f"[patcher] Starting vLLM with: {' '.join(CLI_ARGS)}", flush=True)

# ── Step 4: Must import run_server AFTER parsing, but BEFORE serving ──
from vllm.entrypoints.openai.api_server import run_server
run_server(args)
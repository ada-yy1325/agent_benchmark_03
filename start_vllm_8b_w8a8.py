#!/usr/bin/env python3
"""
Patched vLLM server launcher for Qwen3-8B-Instruct W8A8 (GPQA-Diamond test).
Port 8812, served-model-name Qwen3-8B-W8A8.
Includes maybe_update_config patch for ModelSlim quant config loading.
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
# NOTE: /inspire/ shared filesystem does NOT support symlink resolution.
# Use the real path directly instead of models/Qwen3-8B-W8A8 symlink.
MODEL_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/ZKMatrix/Qwen3-8B-w8a8-full"

# ── Step 3: Patch maybe_update_config ──
import vllm_ascend.quantization.modelslim_config as modelslim_cfg
import vllm_ascend.quantization.utils as quant_utils

_orig_maybe_update_config = modelslim_cfg.AscendModelSlimConfig.maybe_update_config

def _patched_maybe_update_config(self, model_name, hf_config=None, revision=None):
    if self.quant_description:
        return

    from vllm_ascend.quantization.modelslim_config import MODELSLIM_CONFIG_FILENAME
    config_path = quant_utils.get_model_file(MODEL_DIR, MODELSLIM_CONFIG_FILENAME, revision=revision)
    if config_path is not None:
        with open(config_path) as f:
            self.quant_description = json.load(f)
        self._apply_extra_quant_adaptations()
        self._add_kvcache_quant_metadata()
        print(f"[patcher] ✓ Loaded quant config from MODEL_DIR: {config_path}", flush=True)
        return

    print(f"[patcher] ⚠ MODEL_DIR failed, trying model_name: {model_name}", flush=True)
    config_path = quant_utils.get_model_file(model_name, MODELSLIM_CONFIG_FILENAME, revision=revision)
    if config_path is not None:
        with open(config_path) as f:
            self.quant_description = json.load(f)
        self._apply_extra_quant_adaptations()
        self._add_kvcache_quant_metadata()
        print(f"[patcher] ✓ Loaded quant config from (model_name): {config_path}", flush=True)
        return

    # Fallback: list JSON files in MODEL_DIR for debugging
    json_names = []
    if os.path.isdir(MODEL_DIR):
        import glob
        json_files = glob.glob(os.path.join(MODEL_DIR, "*.json"))
        json_names = [os.path.basename(f) for f in json_files]

    from vllm.logger import logger
    logger.error(
        "ModelSlim quantization config not found. "
        "Tried MODEL_DIR='%s' and model_name='%s'. Found JSON files: %s.",
        MODEL_DIR, model_name, json_names if json_names else "N/A",
    )
    raise ValueError(
        f"\n{'=' * 80}\n"
        f"ERROR: ModelSlim Quantization Config Not Found\n"
        f"{'=' * 80}\n\n"
        f"The patcher tried:\n"
        f"  MODEL_DIR: {MODEL_DIR} ({'EXISTS' if os.path.isdir(MODEL_DIR) else 'MISSING'})\n"
        f"  model_name (from vLLM): {model_name}\n"
        f"  hf_config._name_or_path: {hf_config._name_or_path if hf_config else 'N/A'}\n"
        f"\n{'=' * 80}"
    )

modelslim_cfg.AscendModelSlimConfig.maybe_update_config = _patched_maybe_update_config
print("[patcher] ✓ maybe_update_config patched", flush=True)

# ── Step 4: Build CLI args ──
CLI_ARGS = [
    MODEL_DIR,
    '--port', '8812',
    '--max-model-len', '8192',
    '--quantization', 'ascend',
    '--dtype', 'auto',
    '--served-model-name', 'Qwen3-8B-W8A8',
    '--trust-remote-code',
    '--gpu-memory-utilization', '0.9',
    '--enforce-eager',
]

# ── Step 5: Parse args and start vLLM ──
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
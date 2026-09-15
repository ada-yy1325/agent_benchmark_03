#!/usr/bin/env python3
"""
Patched vLLM server launcher.
1. Disables AOTAutogradCache before vLLM loads, to work around
   the AssertionError: expected OutputCode, got _CompiledFxGraph
   bug in PyTorch 2.11 + vLLM-Ascend integration.
2. Fixes maybe_update_config to use hf_config._name_or_path instead
   of model_name (which is incorrectly resolved to a HF repo ID),
   so the quantization config is found from the local model directory.
"""
import sys
import os
import asyncio
import json

# ── Step 1: Disable AOTAutogradCache BEFORE any vLLM import ──────────
import torch._functorch.config as _functorch_cfg
_functorch_cfg._config['enable_autograd_cache'] = False
_functorch_cfg._config['bypass_autograd_cache_key'] = True
print("[patcher] ✓ AOTAutogradCache disabled", flush=True)

# ── Step 2: Patch maybe_update_config ──
#   The bug: vLLM passes model_config.model (= "Qwen/Qwen3-0.6B") as model_name,
#            but the quant config file is in our local model directory.
#   The fix:  Use hf_config._name_or_path when it's a valid local directory.

import vllm_ascend.quantization.modelslim_config as modelslim_cfg
import vllm_ascend.quantization.utils as quant_utils

_orig_maybe_update_config = modelslim_cfg.AscendModelSlimConfig.maybe_update_config

def _patched_maybe_update_config(self, model_name, hf_config=None, revision=None):
    # ── If quant_description is already populated, skip ──
    if self.quant_description:
        return

    # ── Determine the real search path ──
    # Try hf_config._name_or_path first (it's set to the local path by transformers)
    search_path = model_name  # fallback
    if hf_config is not None and hf_config._name_or_path:
        maybe_path = hf_config._name_or_path
        if os.path.isdir(maybe_path):
            search_path = maybe_path
            print(f"[patcher] ✓ Using hf_config._name_or_path: {search_path}", flush=True)
        else:
            # hf_config._name_or_path might be a HF repo ID like "Qwen/Qwen3-0.6B"
            print(f"[patcher] ⚠ hf_config._name_or_path '{maybe_path}' is not a dir, fallback to '{model_name}'", flush=True)
    
    # ── Now try to find the config file using get_model_file ──
    from vllm_ascend.quantization.modelslim_config import MODELSLIM_CONFIG_FILENAME
    config_path = quant_utils.get_model_file(search_path, MODELSLIM_CONFIG_FILENAME, revision=revision)
    
    if config_path is not None:
        with open(config_path) as f:
            self.quant_description = json.load(f)
        self._apply_extra_quant_adaptations()
        self._add_kvcache_quant_metadata()
        print(f"[patcher] ✓ Loaded quant config from: {config_path}", flush=True)
        return
    
    # If still not found, try model_name too (for edge cases)
    if search_path != model_name:
        print(f"[patcher] ⚠ Not found at '{search_path}', trying model_name '{model_name}'", flush=True)
        config_path = quant_utils.get_model_file(model_name, MODELSLIM_CONFIG_FILENAME, revision=revision)
        if config_path is not None:
            with open(config_path) as f:
                self.quant_description = json.load(f)
            self._apply_extra_quant_adaptations()
            self._add_kvcache_quant_metadata()
            print(f"[patcher] ✓ Loaded quant config from (model_name): {config_path}", flush=True)
            return
    
    # ── Nothing found — show helpful error ──
    # Collect diagnostic info
    json_names = []
    if os.path.isdir(search_path):
        import glob
        json_files = glob.glob(os.path.join(search_path, "*.json"))
        json_names = [os.path.basename(f) for f in json_files]
    
    from vllm.logger import logger
    logger.error(
        "ModelSlim quantization config not found for model '%s'. Searched path: %s. Found JSON files: %s.",
        model_name,
        search_path,
        json_names if json_names else "N/A",
    )
    raise ValueError(
        "\n"
        + "=" * 80
        + "\n"
        + "ERROR: ModelSlim Quantization Config Not Found\n"
        + "=" * 80
        + "\n\n"
        + "The patcher tried both models names but could not find the config.\n"
        + f"  model_name (from vLLM): {model_name}\n"
        + f"  hf_config._name_or_path: {hf_config._name_or_path if hf_config else 'N/A'}\n"
        + f"  search_path: {search_path}\n\n"
        + "=" * 80
    )

modelslim_cfg.AscendModelSlimConfig.maybe_update_config = _patched_maybe_update_config
print("[patcher] ✓ maybe_update_config patched to use local path", flush=True)

# ── Step 3: Build CLI args ──
MODEL_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/Qwen3-4B-W8A8"
CLI_ARGS = [
    MODEL_DIR,
    '--port', '8801',
    '--max-model-len', '8192',
    '--quantization', 'ascend',
    '--dtype', 'auto',
    '--served-model-name', 'Qwen3-4B',
]

# ── Step 4: Parse args and start vLLM ──
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args

parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
parser = make_arg_parser(parser)
args = parser.parse_args(CLI_ARGS)
validate_parsed_serve_args(args)
print(f"[patcher] Starting vLLM with: {' '.join(CLI_ARGS)}", flush=True)

# ── Step 5: run_server is async, need event loop ──
from vllm.entrypoints.openai.api_server import run_server
asyncio.run(run_server(args))
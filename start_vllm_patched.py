#!/usr/bin/env python3
"""
Patched vLLM server launcher.
1. Disables AOTAutogradCache before vLLM loads, to work around
   the AssertionError: expected OutputCode, got _CompiledFxGraph
   bug in PyTorch 2.11 + vLLM-Ascend integration.
2. Fixes maybe_update_config to always try MODEL_DIR (the known local
   path) first, bypassing model_name which vLLM incorrectly resolves
   to a HF repo ID instead of the local model directory.
"""
import sys
import os
import asyncio
import json

# ── Step 1: Disable AOTAutogradCache BEFORE any vLLM import ──────────
#   Use proper API (attribute assignment), not direct _config dict mutation.
#   Directly setting _config dict items with plain bools breaks the nested
#   config module structure, causing AttributeError: 'bool' object has no
#   attribute 'hide' during aot_compile.
import torch._functorch.config as _functorch_cfg
_functorch_cfg.enable_autograd_cache = False     # type: ignore[assignment]
_functorch_cfg.bypass_autograd_cache_key = True  # type: ignore[assignment]
print("[patcher] ✓ AOTAutogradCache disabled (via attribute set)", flush=True)

# ── Step 2: Define MODEL_DIR early (used by the patcher below) ──
MODEL_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/Qwen3-4B-W8A8"

# ── Step 3: Patch maybe_update_config ──
#   The bug: vLLM passes model_config.model (= "Qwen/Qwen3-0.6B") as model_name,
#            and hf_config._name_or_path is also polluted.
#   The fix:  Always try MODEL_DIR (the known local path) first.

import vllm_ascend.quantization.modelslim_config as modelslim_cfg
import vllm_ascend.quantization.utils as quant_utils

_orig_maybe_update_config = modelslim_cfg.AscendModelSlimConfig.maybe_update_config

def _patched_maybe_update_config(self, model_name, hf_config=None, revision=None):
    # ── If quant_description is already populated, skip ──
    if self.quant_description:
        return

    # ── Always try the known local path FIRST ──
    from vllm_ascend.quantization.modelslim_config import MODELSLIM_CONFIG_FILENAME
    config_path = quant_utils.get_model_file(MODEL_DIR, MODELSLIM_CONFIG_FILENAME, revision=revision)
    if config_path is not None:
        with open(config_path) as f:
            self.quant_description = json.load(f)
        self._apply_extra_quant_adaptations()
        self._add_kvcache_quant_metadata()
        print(f"[patcher] ✓ Loaded quant config from MODEL_DIR: {config_path}", flush=True)
        return

    # ── Fallback: try model_name ──
    print(f"[patcher] ⚠ MODEL_DIR failed, trying model_name: {model_name}", flush=True)
    config_path = quant_utils.get_model_file(model_name, MODELSLIM_CONFIG_FILENAME, revision=revision)
    if config_path is not None:
        with open(config_path) as f:
            self.quant_description = json.load(f)
        self._apply_extra_quant_adaptations()
        self._add_kvcache_quant_metadata()
        print(f"[patcher] ✓ Loaded quant config from (model_name): {config_path}", flush=True)
        return

    # ── Nothing found — show helpful error ──
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
print("[patcher] ✓ maybe_update_config patched to use MODEL_DIR", flush=True)

# ── Step 4: Build CLI args ──
CLI_ARGS = [
    MODEL_DIR,
    '--port', '8801',
    '--max-model-len', '8192',
    '--quantization', 'ascend',
    '--dtype', 'auto',
    '--served-model-name', 'Qwen3-4B',
]

# ── Step 5: Parse args and start vLLM ──
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args

parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
parser = make_arg_parser(parser)
args = parser.parse_args(CLI_ARGS)
validate_parsed_serve_args(args)

# ── Fix: model_tag → model ──
#   The CLI parser stores the positional arg as model_tag,
#   but api_server.py reads args.model directly.
#   launch.py / serve.py do this mapping, but we bypass those.
if hasattr(args, 'model_tag') and args.model_tag is not None:
    args.model = args.model_tag
if hasattr(args, 'tokenizer_tag') and args.tokenizer_tag is not None:
    # Mirror the same pattern for tokenizer
    args.tokenizer = args.tokenizer_tag

print(f"[patcher] Starting vLLM with: {' '.join(CLI_ARGS)}", flush=True)
print(f"[patcher]   model={args.model}, served_model_name={args.served_model_name}", flush=True)

# ── Step 6: run_server is async, need event loop ──
from vllm.entrypoints.openai.api_server import run_server
asyncio.run(run_server(args))
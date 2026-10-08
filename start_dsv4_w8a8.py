#!/usr/bin/env python3
"""
vLLM server launcher for DeepSeek-V4-Flash-w8a8-mtp.
Follows official A3 single-node deployment from vLLM-Ascend docs §5.1.
TP=8, DP=1, uses 8 of 16 available NPUs.

Includes monkey-patches for:
  - indexer_kv_dtype (supported from additional-config)
  - AOTAutogradCache disabled
"""
import json
import os
import sys
import asyncio

# ═══════════════════════════════════════════════════════════════
# Step 1: Patches BEFORE any vLLM import
# ═══════════════════════════════════════════════════════════════

# 1a. Disable AOTAutogradCache
import torch._functorch.config as _functorch_cfg
_functorch_cfg.enable_autograd_cache = False
_functorch_cfg.bypass_autograd_cache_key = True
print("[launcher] ✓ AOTAutogradCache disabled", flush=True)

# 1b. Monkey-patch DeepseekV4Indexer to support indexer_kv_dtype
import vllm.models.deepseek_v4.attention as dsv4_attn

_orig_indexer_init = dsv4_attn.DeepseekV4Indexer.__init__

def _patched_indexer_init(self, vllm_config, *args, **kwargs):
    """Patched __init__ that reads indexer_kv_dtype from additional_config."""
    # Call original init first (sets scale_fmt="ue8m0")
    _orig_indexer_init(self, vllm_config, *args, **kwargs)

    # Read indexer_kv_dtype from additional_config
    if vllm_config is not None and vllm_config.additional_config:
        indexer_kv_dtype = vllm_config.additional_config.get("indexer_kv_dtype", "")
        if indexer_kv_dtype == "int8":
            self.scale_fmt = "s8"
            self.quant_block_size = 64
            print(f"[launcher] ⚡ indexer_kv_dtype=int8 → scale_fmt={self.scale_fmt}, "
                  f"quant_block_size={self.quant_block_size}", flush=True)
        elif indexer_kv_dtype == "fp8":
            # Already default
            pass
        elif indexer_kv_dtype:
            print(f"[launcher] ⚠ Unknown indexer_kv_dtype={indexer_kv_dtype}, "
                  f"keeping default scale_fmt={self.scale_fmt}", flush=True)

dsv4_attn.DeepseekV4Indexer.__init__ = _patched_indexer_init
print("[launcher] ✓ DeepseekV4Indexer patched for indexer_kv_dtype support", flush=True)

# ═══════════════════════════════════════════════════════════════
# Step 2: Set environment variables (official A3 config)
# ═══════════════════════════════════════════════════════════════
os.environ.setdefault("OMP_PROC_BIND", "false")
os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HCCL_BUFFSIZE", "1024")
os.environ.setdefault("TASK_QUEUE_ENABLE", "1")
os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "AIV")
os.environ.setdefault("VLLM_ASCEND_ENABLE_FLASHCOMM1", "1")  # required by enable_dsa_cp

# ═══════════════════════════════════════════════════════════════
# Step 3: Define MODEL_DIR
# ═══════════════════════════════════════════════════════════════
MODEL_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-mtp"

# ═══════════════════════════════════════════════════════════════
# Step 4: Build CLI args (official A3 single-node, adapted for vLLM 0.22.1)
# ═══════════════════════════════════════════════════════════════
ADDITIONAL_CONFIG = {
    "ascend_compilation_config": {
        "enable_npugraph_ex": True,
        "enable_static_kernel": False,
    },
    "enable_cpu_binding": True,
    "enable_dsa_cp": True,
    "multistream_overlap_shared_expert": True,
    # Official config uses --attention_config.indexer_kv_dtype int8
    # Since vLLM 0.22.1 doesn't support this in AttentionConfig, we pass it
    # through additional-config and read it in the monkey-patch above.
    "indexer_kv_dtype": "int8",
}

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
    '--additional-config', json.dumps(ADDITIONAL_CONFIG),
]

# ═══════════════════════════════════════════════════════════════
# Step 5: Start vLLM
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"[launcher] MODEL_DIR: {MODEL_DIR}", flush=True)
    print(f"[launcher] Additional config: {json.dumps(ADDITIONAL_CONFIG, indent=2)}", flush=True)
    print(f"[launcher] Starting vLLM with:", flush=True)
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
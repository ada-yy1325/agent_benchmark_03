#!/usr/bin/env python3
"""
vLLM server launcher for DeepSeek-V4-Flash-w8a8-mtp.
Follows official A3 single-node deployment from vLLM-Ascend docs §5.1.
TP=8, DP=1, uses 8 of 16 available NPUs.

NOTE:
  - --attention_config.indexer_kv_dtype (dot notation) not supported in vLLM 0.22.1;
    AttentionConfig does not have this field.
  - --speculative-config, --compilation-config, --additional-config take JSON strings.
"""
import os
import sys
import asyncio

# ── Step 1: Set environment variables (official A3 config) ──
os.environ.setdefault("OMP_PROC_BIND", "false")
os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HCCL_BUFFSIZE", "1024")
os.environ.setdefault("TASK_QUEUE_ENABLE", "1")
os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "AIV")
os.environ.setdefault("VLLM_ASCEND_ENABLE_FLASHCOMM1", "1")  # required by enable_dsa_cp

# ── Step 2: Define MODEL_DIR ──
MODEL_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-mtp"

# ── Step 3: Build CLI args (official A3 single-node, adapted for vLLM 0.22.1) ──
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

# ── Step 4: Start vLLM ──
if __name__ == "__main__":
    print(f"[launcher] MODEL_DIR: {MODEL_DIR}", flush=True)
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
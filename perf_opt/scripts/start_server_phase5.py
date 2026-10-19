#!/usr/bin/env python3
"""
Phase 5 flexible vLLM server launcher for DeepSeek-V4-Flash-W8A8.
All switches controllable via CLI for single-variable A/B testing.

Usage examples:
  # Baseline (5.0): MTP=OFF, prefix=OFF, cudagraph=FULL_DECODE_ONLY, all additional ON
  python3 start_server_phase5.py --port 8000 --mtp off

  # Prefix caching ON (5.1)
  python3 start_server_phase5.py --port 8000 --mtp off --prefix-caching

  # Eager mode (5.2)
  python3 start_server_phase5.py --port 8000 --mtp off --graph-mode eager

  # NPU graph (5.2)
  python3 start_server_phase5.py --port 8000 --mtp off --graph-mode npugraph

  # Disable CPU binding (5.3)
  python3 start_server_phase5.py --port 8000 --mtp off --cpu-binding false

  # KV cache fp8 (5.4)
  python3 start_server_phase5.py --port 8000 --mtp off --kv-cache-dtype fp8
"""
import os, sys, asyncio, argparse, json

# ── AOTAutogradCache ──
import torch._functorch.config as _functorch_cfg
_functorch_cfg.enable_autograd_cache = False
_functorch_cfg.bypass_autograd_cache_key = True

# ── Env vars ──
os.environ.setdefault("OMP_PROC_BIND", "false")
os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HCCL_BUFFSIZE", "1024")
os.environ.setdefault("TASK_QUEUE_ENABLE", "1")
os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "AIV")
os.environ.setdefault("VLLM_ASCEND_ENABLE_FLASHCOMM1", "1")

parser = argparse.ArgumentParser(description="Phase 5 flexible vLLM launcher")
# ── Core params ──
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--batched", type=int, default=8192)
parser.add_argument("--max-seqs", type=int, default=64)
parser.add_argument("--tp", type=int, default=8)
parser.add_argument("--mtp", type=str, default="off",
                    choices=["off", "0", "1", "2", "3"])
parser.add_argument("--enforce-eager", action="store_true")

# ── 5.1 Prefix caching ──
parser.add_argument("--prefix-caching", action="store_true", default=False)

# ── 5.2 Graph mode ──
parser.add_argument("--graph-mode", type=str, default="cudagraph",
                    choices=["eager", "default", "cudagraph", "npugraph"])

# ── 5.3 Additional config switches ──
parser.add_argument("--cpu-binding", type=str, default="true", choices=["true", "false"])
parser.add_argument("--dsa-cp", type=str, default="true", choices=["true", "false"])
parser.add_argument("--multistream-overlap", type=str, default="true", choices=["true", "false"])
parser.add_argument("--static-kernel", type=str, default="false", choices=["true", "false"])

# ── 5.4 KV cache dtype ──
parser.add_argument("--kv-cache-dtype", type=str, default="auto")

# ── Extra passthrough ──
parser.add_argument("--extra-vllm-args", type=str, default="")

args_cmd = parser.parse_args()
MODEL_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-self"

CLI_ARGS = [
    MODEL_DIR,
    "--host", "0.0.0.0",
    "--port", str(args_cmd.port),
    "--max-model-len", "133120",
    "--max-num-batched-tokens", str(args_cmd.batched),
    "--served-model-name", "dsv4",
    "--gpu-memory-utilization", "0.9",
    "--max-num-seqs", str(args_cmd.max_seqs),
    "--data-parallel-size", "1",
    "--tensor-parallel-size", str(args_cmd.tp),
    "--enable-expert-parallel",
    "--tokenizer-mode", "deepseek_v4",
    "--tool-call-parser", "deepseek_v4",
    "--enable-auto-tool-choice",
    "--reasoning-parser", "deepseek_v4",
    "--model-loader-extra-config", '{"enable_multithread_load": true, "num_threads": 128}',
    "--quantization", "ascend",
    "--block-size", "128",
]

# ── 5.1 Prefix caching ──
if args_cmd.prefix_caching:
    CLI_ARGS.append("--enable-prefix-caching")
    print("[launcher] prefix_caching: ENABLED", flush=True)
else:
    CLI_ARGS.append("--no-enable-prefix-caching")
    print("[launcher] prefix_caching: DISABLED", flush=True)

# ── 5.2 Graph mode ──
graph_mode = args_cmd.graph_mode
additional_config = {}

if graph_mode == "eager":
    CLI_ARGS.append("--enforce-eager")
    print("[launcher] graph_mode: eager (--enforce-eager)", flush=True)
elif graph_mode == "default":
    print("[launcher] graph_mode: default (no enforce-eager, no compilation)", flush=True)
elif graph_mode == "cudagraph":
    CLI_ARGS.extend([
        "--compilation-config", '{"cudagraph_mode": "FULL_DECODE_ONLY"}',
    ])
    print("[launcher] graph_mode: cudagraph (FULL_DECODE_ONLY)", flush=True)
elif graph_mode == "npugraph":
    CLI_ARGS.extend([
        "--compilation-config", '{"cudagraph_mode": "FULL_DECODE_ONLY"}',
    ])
    additional_config["ascend_compilation_config"] = {
        "enable_npugraph_ex": True,
        "enable_static_kernel": args_cmd.static_kernel == "true",
    }
    print("[launcher] graph_mode: npugraph (cudagraph + npugraph_ex=True)", flush=True)

# ── 5.3 Additional config ──
if "ascend_compilation_config" not in additional_config:
    additional_config["ascend_compilation_config"] = {
        "enable_npugraph_ex": False,
        "enable_static_kernel": args_cmd.static_kernel == "true",
    }

additional_config["enable_cpu_binding"] = args_cmd.cpu_binding == "true"
additional_config["enable_dsa_cp"] = args_cmd.dsa_cp == "true"
additional_config["multistream_overlap_shared_expert"] = args_cmd.multistream_overlap == "true"

CLI_ARGS.extend([
    "--additional-config",
    json.dumps(additional_config, separators=(",", ":")),
])
print(f"[launcher] additional_config: {json.dumps(additional_config)}", flush=True)

# ── 5.4 KV cache dtype ──
if args_cmd.kv_cache_dtype != "auto":
    CLI_ARGS.extend(["--kv-cache-dtype", args_cmd.kv_cache_dtype])
    print(f"[launcher] kv_cache_dtype: {args_cmd.kv_cache_dtype}", flush=True)

# ── MTP speculative config ──
mtp = args_cmd.mtp
if mtp in ("off", "0"):
    print("[launcher] MTP: OFF (no speculative config)", flush=True)
else:
    n_spec = int(mtp)
    spec_cfg = {
        "num_speculative_tokens": n_spec,
        "method": "mtp",
        "enforce_eager": True,
    }
    print(f"[launcher] MTP: spec={n_spec}, enforce_eager=True", flush=True)
    CLI_ARGS.extend([
        "--speculative-config", json.dumps(spec_cfg, separators=(",", ":")),
    ])

# ── Extra passthrough args ──
if args_cmd.extra_vllm_args:
    extra = args_cmd.extra_vllm_args.split()
    CLI_ARGS.extend(extra)
    print(f"[launcher] extra_vllm_args: {extra}", flush=True)

# ── Summary ──
print(f"[launcher] TP={args_cmd.tp} port={args_cmd.port} "
      f"batched={args_cmd.batched} max_seqs={args_cmd.max_seqs}", flush=True)
devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "NOT SET")
print(f"[launcher] ASCEND_RT_VISIBLE_DEVICES={devices}", flush=True)
print(f"[launcher] Full CLI:", " ".join(CLI_ARGS), flush=True)

if __name__ == "__main__":
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args

    p = FlexibleArgumentParser(description="vLLM API server (phase5)")
    p = make_arg_parser(p)
    a = p.parse_args(CLI_ARGS)
    validate_parsed_serve_args(a)

    if hasattr(a, 'model_tag') and a.model_tag is not None:
        a.model = a.model_tag
    if hasattr(a, 'tokenizer_tag') and a.tokenizer_tag is not None:
        a.tokenizer = a.tokenizer_tag

    from vllm.entrypoints.openai.api_server import run_server
    asyncio.run(run_server(a))
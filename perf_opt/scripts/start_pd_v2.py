#!/usr/bin/env python3
"""
PD Separation quick test for DeepSeek-V4-Flash-W8A8 on Ascend NPU.
Runs two TP8 instances (prefill on cards 0-7, decode on 8-15) 
with KV transfer via AscendStoreConnector + MemcacheBackend.

Usage (run on remote via tmux):
  python3 start_pd_test.py
"""
import os, sys, asyncio, json, time, subprocess, urllib.request

# Suppress functorch cache
import torch._functorch.config as _fc
_fc.enable_autograd_cache = False
_fc.bypass_autograd_cache_key = True

MODEL_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-npu16"

KV_TRANSFER_CONFIG = {
    "kv_connector": "AscendStoreConnector",
    "kv_role": "PLACEHOLDER",  # "kv_producer" or "kv_consumer"
    "kv_parallel_size": 2,
    "kv_ip": "127.0.0.1",
    "kv_port": 14579,
    "kv_rank": 0,
    "kv_buffer_size": 5e8,
    "kv_load_failure_policy": "fail",
    "kv_connector_extra_config": {
        "backend": "shm",
        "lookup_rpc_port": 14579,
    },
}

def build_args(port, is_prefill):
    cfg = dict(KV_TRANSFER_CONFIG)
    cfg["kv_role"] = "kv_producer" if is_prefill else "kv_consumer"
    cfg["kv_rank"] = 0 if is_prefill else 1
    if not is_prefill:
        cfg["kv_connector_extra_config"]["consumer_is_to_load"] = True

    args = [
        MODEL_DIR, "--host", "0.0.0.0", "--port", str(port),
        "--max-model-len", "133120", "--max-num-batched-tokens", "8192",
        "--served-model-name", "dsv4", "--gpu-memory-utilization", "0.9",
        "--max-num-seqs", "64", "--data-parallel-size", "1",
        "--tensor-parallel-size", "8", "--enable-expert-parallel",
        "--quantization", "ascend", "--block-size", "128",
        "--compilation-config", '{"cudagraph_mode": "FULL_DECODE_ONLY"}',
        "--kv-transfer-config", json.dumps(cfg, separators=(",", ":")),
        "--additional-config",
        '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":true},"enable_cpu_binding":true,"enable_dsa_cp":true,"multistream_overlap_shared_expert":true}',
    ]
    
    if is_prefill:
        # tokens-only mode: NO tokenizer available, so skip tokenizer-dependent args
        args.append("--tokens-only")
    else:
        # Decode: needs tokenizer for text output
        args.extend([
            "--tokenizer-mode", "deepseek_v4",
            "--tool-call-parser", "deepseek_v4",
            "--enable-auto-tool-choice",
            "--reasoning-parser", "deepseek_v4",
            "--speculative-config",
            '{"num_speculative_tokens": 1, "method": "mtp", "enforce_eager": true}',
        ])
    
    return args


def start_vllm(name, devices, args, logfile):
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    # Write args to temp file for safe passing
    args_file = os.path.join(os.path.dirname(logfile), f"{name}_args.json")
    with open(args_file, "w") as f:
        json.dump(args, f)
    script = f"""#!/bin/bash
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024 TASK_QUEUE_ENABLE=1 HCCL_OP_EXPANSION_MODE=AIV VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export ASCEND_RT_VISIBLE_DEVICES={devices}
export _PD_ARGS_FILE={args_file}
cd /inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test
exec python3 {__file__} 2>&1 | tee {logfile}
"""
    script_path = os.path.join(os.path.dirname(logfile), f"_{name}_launch.sh")
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    os.system(f"tmux kill-session -t {name} 2>/dev/null")
    os.system(f"tmux new-session -d -s {name} bash {script_path}")
    print(f"[launcher] Started tmux '{name}' (devices={devices})")


def wait_healthy(port, timeout=900):
    url = f"http://127.0.0.1:{port}/health"
    print(f"[launcher] Waiting for :{port}/health...")
    for i in range(1, timeout + 1):
        try:
            urllib.request.urlopen(url, timeout=3)
            print(f"[launcher] :{port} healthy ({i}s)")
            return True
        except Exception:
            if i % 60 == 0:
                print(f"[launcher] :{port} ... {i}s")
            time.sleep(1)
    return False


def serve_mode():
    """Run actual vLLM server with args from temp file."""
    args_file = os.environ.get("_PD_ARGS_FILE", "")
    if not args_file or not os.path.exists(args_file):
        return False
    with open(args_file, "r") as f:
        cli_args = json.load(f)
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    p = FlexibleArgumentParser(description="vLLM PD")
    p = make_arg_parser(p)
    a = p.parse_args(cli_args)
    validate_parsed_serve_args(a)
    if hasattr(a, "model_tag") and a.model_tag is not None:
        a.model = a.model_tag
    if hasattr(a, "tokenizer_tag") and a.tokenizer_tag is not None:
        a.tokenizer = a.tokenizer_tag
    from vllm.entrypoints.openai.api_server import run_server
    asyncio.run(run_server(a))
    return True


def main():
    if serve_mode():
        return

    # Main mode: launch both servers
    os.system("pkill -9 -f 'VLLM::' 2>/dev/null; sleep 2")
    for p in [8000, 8001]:
        os.system(f"fuser -k {p}/tcp 2>/dev/null")

    print("=" * 60)
    print("PD Separation v2: Prefill(0-7) + Decode(8-15)")
    print("=" * 60)

    # Start decode first (consumer), then prefill (producer)
    d_args = build_args(8001, is_prefill=False)
    start_vllm("pd_decode", "8,9,10,11,12,13,14,15", d_args, "/tmp/pd_test/decode_v2.log")
    time.sleep(3)
    
    p_args = build_args(8000, is_prefill=True)
    start_vllm("pd_prefill", "0,1,2,3,4,5,6,7", p_args, "/tmp/pd_test/prefill_v2.log")

    print("\nServers started. Waiting 15s then checking health...")
    time.sleep(15)
    
    healthy_decode = wait_healthy(8001, 60)
    healthy_prefill = wait_healthy(8000, 60)
    
    if healthy_decode and healthy_prefill:
        print("\n✅ Both servers healthy! PD separation is RUNNING.")
    elif healthy_decode:
        print("\n⚠️ Decode healthy but prefill not yet ready (graph may still be compiling)")
    elif healthy_prefill:
        print("\n⚠️ Prefill healthy but decode not yet ready")
    else:
        print("\n⏳ Servers still starting (graph compilation ~10-25min). Check later:")
        print("  tmux attach -t pd_decode")
        print("  tmux attach -t pd_prefill")
        print("  curl http://127.0.0.1:8000/health")
        print("  curl http://127.0.0.1:8001/health")

if __name__ == "__main__":
    main()
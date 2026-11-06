#!/usr/bin/env python3
"""
PD Mooncake: Prefill(0-7) + Decode(8-15) dual-instance launcher.
Supports serve mode (_PD_ARGS_FILE set) and orchestrator mode (--serial).

Reconstructed 2026-11-06 from patches & arg dumps. Zero LD pollution:
no LD_PRELOAD, no LD_LIBRARY_PATH mooncake (connector preloads its own deps via ctypes).
"""
import argparse, asyncio, json, os, sys, time

PROJECT = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test"
MODEL = f"{PROJECT}/models/DeepSeek-V4-Flash-w8a8-npu16"
SCRIPTS = f"{PROJECT}/perf_opt/scripts"
TMP = "/tmp/pd_test"

# ── Mooncake environment variables (unified for both sides) ────────────────
MC_ENV = {
    "MC_TCP_ENABLE_CONNECTION_POOL": "0",
    "MC_METADATA_CLUSTER_ID": "0",
    "MC_NUM_COMP_CHANNELS_PER_CTX": "2",
    "MC_AUTO_GID_MAX_RETRIES": "3",
    "MC_TE_METADATA_REFRESH_INTERVAL_SECONDS": "1",
    "MC_HANDSHAKE_LISTEN_BACKLOG": "128",
    "MC_HANDSHAKE_CONNECT_TIMEOUT": "10",
    "MC_RDMA_RAIL_PAUSE_SECONDS": "5",
    "MC_RDMA_WORKER_IDLE_SPIN_US": "0",
    "MC_ENABLE_HCA_PEER_AFFINITY": "0",
    "MC_LOG_RDMA_SLICE_AFFINITY": "0",
    "MC_TRACK_RDMA_POSTED_SLICES": "0",
    "MC_EFA_HOMOGENEOUS_PEERS": "0",
    "MC_EFA_CQ_DRAIN_TIMEOUT_MS": "1000",
    "MC_RDMA_NOTIFY_RECV_COUNT": "4",
    "MC_RDMA_NOTIFY_BUFFER_SIZE": "4096",
    "MC_RDMA_NOTIFY_MAX_PENDING_SENDS": "16",
    "MC_RDMA_NOTIFY_OOB_FALLBACK": "1",
    "MC_TRANSFER_TIMEOUT": "30",
    "MC_CUSTOM_TOPO_JSON": "",
}

# ── Common additional-config JSON ──────────────────────────────────────────
ADDITIONAL_CONFIG = json.dumps({
    "ascend_compilation_config": {
        "enable_npugraph_ex": True,
        "enable_static_kernel": False,
    },
    "enable_cpu_binding": True,
    "enable_dsa_cp": True,
    "multistream_overlap_shared_expert": True,
}, separators=(",", ":"))

# ── Common kv_connector_extra_config ───────────────────────────────────────
KV_EXTRA = {
    "mooncake_protocol": "ascend",
    "prefill": {"dp_size": 1, "tp_size": 8},
    "decode": {"dp_size": 1, "tp_size": 8},
}


def build_kv_config(role: str, rank: int) -> str:
    """Build kv-transfer-config JSON for producer or consumer."""
    cfg = {
        "kv_connector": "MooncakeConnector",
        "kv_role": role,
        "kv_parallel_size": 2,
        "kv_ip": "127.0.0.1",
        "kv_port": 38000,
        "kv_rank": rank,
        "kv_buffer_size": 8_000_000_000.0,
        "kv_load_failure_policy": "fail",
        "kv_connector_extra_config": KV_EXTRA,
    }
    return json.dumps(cfg, separators=(",", ":"))


def build_args(prefill: bool, port: int) -> list:
    """Build CLI args for vllm serve."""
    role = "kv_producer" if prefill else "kv_consumer"
    rank = 0 if prefill else 1
    args = [
        MODEL,
        "--host", "0.0.0.0", "--port", str(port),
        "--max-model-len", "133120",
        "--max-num-batched-tokens", "8192",
        "--served-model-name", "dsv4",
        "--gpu-memory-utilization", "0.9",
        "--max-num-seqs", "64",
        "--data-parallel-size", "1",
        "--tensor-parallel-size", "8",
        "--enable-expert-parallel",
        "--quantization", "ascend",
        "--block-size", "128",
        "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
        "--kv-transfer-config", build_kv_config(role, rank),
        "--additional-config", ADDITIONAL_CONFIG,
        "--tokenizer-mode", "deepseek_v4",
        "--tool-call-parser", "deepseek_v4",
        "--enable-auto-tool-choice",
        "--reasoning-parser", "deepseek_v4",
    ]
    if prefill:
        # Prefill: enforce-eager + MTP=1 (speculative)
        args += [
            "--enforce-eager",
            "--speculative-config",
            json.dumps({
                "num_speculative_tokens": 1,
                "method": "mtp",
                "enforce_eager": True,
            }, separators=(",", ":")),
        ]
    return args


def generate_shell_script(name, devices, args_file, logfile):
    """Generate a self-contained bash launch script."""
    lines = ["#!/bin/bash"]
    common_env = [
        "OMP_PROC_BIND=false", "OMP_NUM_THREADS=10",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
    ]
    lines.append("export " + " ".join(common_env))
    lines.append("export HCCL_INTRA_ROCE_ENABLE=1")
    lines.append("export ASCEND_BUFFER_POOL=4:8")
    mc_vars = " ".join(f"{k}={v}" for k, v in MC_ENV.items())
    lines.append(f"export {mc_vars}")
    lines.append(
        "export HCCL_BUFFSIZE=1024 TASK_QUEUE_ENABLE=1 "
        "HCCL_OP_EXPANSION_MODE=AIV VLLM_ASCEND_ENABLE_FLASHCOMM1=1"
    )
    lines.append(f"export ASCEND_RT_VISIBLE_DEVICES={devices}")
    lines.append(f"export _PD_ARGS_FILE={args_file}")
    lines.append(f"cd {PROJECT}")
    launcher = f"{SCRIPTS}/start_pd_mooncake.py"
    lines.append(f"exec python3 {launcher} 2>&1 | tee {logfile}")
    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════════
# Serve mode (called by the shell script with _PD_ARGS_FILE set)
# ═══════════════════════════════════════════════════════════════════════════
async def serve_mode():
    """Read args from _PD_ARGS_FILE and launch vllm API server."""
    args_file = os.environ.get("_PD_ARGS_FILE", "")
    if not args_file:
        print("[serve] ERROR: _PD_ARGS_FILE not set", file=sys.stderr)
        sys.exit(1)
    with open(args_file) as f:
        cli_args = json.load(f)
    print(f"[serve] Args loaded from {args_file}: {len(cli_args)} tokens")
    from vllm.entrypoints.openai.api_server import main as vllm_main
    old_argv = sys.argv
    sys.argv = ["vllm", "serve"] + cli_args
    try:
        await vllm_main()
    finally:
        sys.argv = old_argv


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator mode (--serial)
# ═══════════════════════════════════════════════════════════════════════════
def orchestrator_mode():
    """Kill previous, generate scripts, start decode then prefill via tmux."""
    print("=" * 60)
    print("PD Separation v2: Prefill(0-7) + Decode(8-15)")
    print("Mode: SERIAL (decode first, then prefill)")
    print("=" * 60)

    # Cleanup
    os.system("pkill -9 -f 'VLLM::' 2>/dev/null; sleep 2")
    os.system("fuser -k 8000/tcp 2>/dev/null; fuser -k 8001/tcp 2>/dev/null")
    os.makedirs(TMP, exist_ok=True)

    # Build args
    decode_args = build_args(prefill=False, port=8001)
    prefill_args = build_args(prefill=True, port=8000)

    # Write arg files
    decode_args_file = f"{TMP}/pd_decode_args.json"
    prefill_args_file = f"{TMP}/pd_prefill_args.json"
    with open(decode_args_file, "w") as f:
        json.dump(decode_args, f)
    with open(prefill_args_file, "w") as f:
        json.dump(prefill_args, f)

    # Generate shell scripts
    decode_script = generate_shell_script(
        "pd_decode", "8,9,10,11,12,13,14,15",
        decode_args_file, f"{TMP}/decode_v2.log",
    )
    prefill_script = generate_shell_script(
        "pd_prefill", "0,1,2,3,4,5,6,7",
        prefill_args_file, f"{TMP}/prefill_v2.log",
    )
    decode_sh = f"{TMP}/_pd_decode_launch.sh"
    prefill_sh = f"{TMP}/_pd_prefill_launch.sh"
    for path, content in [(decode_sh, decode_script), (prefill_sh, prefill_script)]:
        with open(path, "w") as f:
            f.write(content)
        os.chmod(path, 0o755)

    # Start decode
    os.system("tmux kill-server 2>/dev/null; sleep 1")
    os.system("tmux new-session -d -s pd_decode bash")
    time.sleep(0.5)
    os.system(f"tmux send-keys -t pd_decode 'bash {decode_sh}' Enter")
    print(f"\n[launcher] Started tmux 'pd_decode' (devices=8,9,10,11,12,13,14,15)")

    # Wait for decode health
    print("\n[serial] Waiting for decode instance to be healthy (up to 30 min)...")
    if not wait_health(8001, timeout=1800):
        print("[ERROR] Decode failed to become healthy. Check decode_v2.log")
        sys.exit(1)

    # Start prefill
    os.system("tmux new-session -d -s pd_prefill bash")
    time.sleep(0.5)
    os.system(f"tmux send-keys -t pd_prefill 'bash {prefill_sh}' Enter")
    print(f"\n[launcher] Started tmux 'pd_prefill' (devices=0,1,2,3,4,5,6,7)")

    # Wait for prefill health
    print("\n[serial] Waiting for prefill instance to be healthy (up to 30 min)...")
    if not wait_health(8000, timeout=1800):
        print("[ERROR] Prefill failed to become healthy. Check prefill_v2.log")
        sys.exit(1)

    print("\n✅ PD Separation v2: both instances healthy!")
    print("   Prefill: http://127.0.0.1:8000 (NPU 0-7, producer)")
    print("   Decode:  http://127.0.0.1:8001 (NPU 8-15, consumer)")


def wait_health(port, timeout=1800):
    """Poll :port/health until 200 or timeout."""
    import urllib.request
    import urllib.error
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=5
            )
            if resp.status == 200:
                print(f"[launcher] :{port} HEALTHY ✓")
                return True
        except (urllib.error.URLError, OSError):
            pass
        remaining = max(0, int(deadline - time.time()))
        print(f"[launcher] :{port} ... {remaining}s")
        time.sleep(10)
    return False


def main():
    parser = argparse.ArgumentParser(description="PD Mooncake launcher")
    parser.add_argument("--serial", action="store_true",
                        help="Orchestrator: start decode then prefill via tmux")
    ns = parser.parse_args()

    if os.environ.get("_PD_ARGS_FILE"):
        asyncio.run(serve_mode())
    elif ns.serial:
        orchestrator_mode()
    else:
        print("Usage: --serial (orchestrator) or set _PD_ARGS_FILE (serve mode)")
        sys.exit(1)


if __name__ == "__main__":
    main()
    common_env = [
        "OMP_PROC_BIND=false", "OMP_NUM_THREADS=10",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
    ]
    lines.append("export " + " ".join(common_env))
    lines.append("export HCCL_INTRA_ROCE_ENABLE=1")
    lines.append("export ASCEND_BUFFER_POOL=4:8")
    mc_vars = " ".join(f"{k}={v}" for k, v in MC_ENV.items())
    lines.append(f"export {mc_vars}")
    lines.append(
        "export HCCL_BUFFSIZE=1024 TASK_QUEUE_ENABLE=1 "
        "HCCL_OP_EXPANSION_MODE=AIV VLLM_ASCEND_ENABLE_FLASHCOMM1=1"
    )
    lines.append(f"export ASCEND_RT_VISIBLE_DEVICES={devices}")
    lines.append(f"export _PD_ARGS_FILE={args_file}")
    lines.append(f"cd {PROJECT}")
    launcher = f"{SCRIPTS}/start_pd_mooncake.py"
    lines.append(f"exec python3 {launcher} 2>&1 | tee {logfile}")
    return "\n".join(lines) + "\n"
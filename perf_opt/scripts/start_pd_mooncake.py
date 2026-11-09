#!/usr/bin/env python3
"""
PD Mooncake: Prefill(0-7) + Decode(8-15) dual-instance launcher.
Supports serve mode (_PD_ARGS_FILE set) and orchestrator mode (--serial).

Environment: 5 conditions required for stability (see PD_HANDOVER_1106 §5):
  (1) mooncake dir: 51 core .so files, 2 CANN dupes in bak_libs/
  (2) LD_LIBRARY_PATH suffix: appends mooncake dir (system CANN priority first)
  (3) ALL MC_* env vars set — each plugs a getenv()→nullptr→basic_string crash
  (4) NO LD_PRELOAD (bt_hook2.so lives in /tmp and survives container rebuild)
  (5) Connector preload patch in mooncake_connector.py (image-level, not git)

Reconstructed 2026-11-06, MC_ENV restored 2026-11-06 from engine binary analysis.
"""
import argparse, json, os, sys, time

PROJECT = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test"
MODEL = f"{PROJECT}/models/DeepSeek-V4-Flash-w8a8-npu16"
SCRIPTS = f"{PROJECT}/perf_opt/scripts"
TMP = "/tmp/pd_test"

# ── Mooncake environment variables ─────────────────────────────────────────
# ALL MC_* vars must be set: mooncake C++ engine calls std::getenv() without
# null-check → std::string(nullptr) → basic_string::_S_construct null not valid
# These were reconstructed from the engine binary's getenv strings (11-06).
MC_ENV = {
    # ── TCP / Networking (loopback-only, no RDMA/IB/EFA) ──
    # NOTE: MC_FORCE_TCP=1 causes Ascend Direct transport to be SKIPPED
    # (see transfer_engine_impl.cpp:240). In TCP mode, the engine uses
    # posix_memalign for memory allocation and handles NPU memory via
    # the 11-02 connector patches (untyped_storage dedup + data-length split).
    # This was verified working on 2026-11-02 (D prompt throughput=0.0).
    #
    # Without MC_FORCE_TCP, engine installs Ascend Direct transport which
    # requires per-worker HCCL memory registration → fails with error 103900
    # (HcclRegisterGlobalMem overlap) in multi-process TP=8 setup.
    "MC_FORCE_TCP": "1",
    "MC_TCP_ENABLE_CONNECTION_POOL": "0",
    "MC_TCP_PROTO": "tcp",
    "MC_TCP_BIND_ADDRESS": "127.0.0.1",
    "MC_TCP_LANES_PER_PEER": "1",
    "MC_TCP_MAX_CONNECTIONS_PER_PEER": "16",
    "MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER": "16",
    "MC_TCP_MAX_QUEUED_BYTES_PER_PEER": "134217728",
    "MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER": "4096",
    "MC_TCP_ADMISSION_TIMEOUT_MS": "30000",
    "MC_TCP_PROGRESS_TIMEOUT_SEC": "30",
    "MC_TCP_STATUS_TIMEOUT_SEC": "300",
    "MC_TCP_SLICE_SIZE": "1048576",
    "MC_NUM_TCP_IO_THREADS": "4",
    "MC_SLICE_SIZE": "1048576",
    "MC_SLICE_TIMEOUT": "30",
    "MC_RETRY_CNT": "10",
    "MC_FRAGMENT_RATIO": "1.0",
    "MC_MIN_REG_SIZE": "4096",
    # ── RDMA (disabled — no RDMA NICs) ──
    "MC_FORCE_SHM": "0",
    "MC_FORCE_HCA": "",
    "MC_GID_INDEX": "3",
    "MC_IB_PORT": "1",
    "MC_IB_SL": "0",
    "MC_IB_TC": "0",
    "MC_MTU": "4096",
    "MC_PKEY_INDEX": "0",
    "MC_MAX_MR_SIZE": "68719476736",
    "MC_MAX_CQE_PER_CTX": "4096",
    "MC_MAX_EP_PER_CTX": "256",
    "MC_MAX_INLINE": "256",
    "MC_MAX_SGE": "4",
    "MC_MAX_WR": "4096",
    "MC_NUM_COMP_CHANNELS_PER_CTX": "1",
    "MC_NUM_CQ_PER_CTX": "1",
    "MC_NUM_QP_PER_EP": "1",
    "MC_WORKERS_PER_CTX": "2",
    "MC_IB_PCI_RELAXED_ORDERING": "0",
    "MC_RDMA_BIND_ADDRESS": "",
    "MC_RDMA_DATA_DIRECT": "0",
    "MC_RDMA_DATA_DIRECT_PAIRING": "",
    "MC_RDMA_NOTIFY_BUFFER_SIZE": "2097152",
    "MC_RDMA_NOTIFY_CONNECT_TIMEOUT_MS": "30000",
    "MC_RDMA_NOTIFY_ENABLED": "0",
    "MC_RDMA_NOTIFY_MAX_PENDING_SENDS": "4096",
    "MC_RDMA_NOTIFY_OOB_FALLBACK": "1",
    "MC_RDMA_NOTIFY_RECV_COUNT": "4096",
    "MC_RDMA_RAIL_PAUSE_SECONDS": "300",
    "MC_RDMA_SIGNAL_INTERVAL": "8192",
    "MC_RDMA_WORKER_IDLE_SPIN_US": "0",
    "MC_USE_RDMA_TWOSIDED": "1",
    "MC_TRACK_RDMA_POSTED_SLICES": "0",
    "MC_PATH_ROUNDROBIN": "1",
    # ── EFA / AWS (disabled) ──
    "MC_EFA_CQ_DRAIN_TIMEOUT_MS": "10000",
    "MC_EFA_CQ_THREADS": "4",
    "MC_EFA_HOMOGENEOUS_PEERS": "0",
    "MC_EFA_NIC_SELECTION": "",
    "MC_EFA_STALL_MIN_MS": "100",
    # ── Transfer Engine ──
    "MC_METADATA_CLUSTER_ID": "",
    "MC_ENDPOINT_STORE_TYPE": "",
    "MC_ENABLE_DEST_DEVICE_AFFINITY": "0",
    "MC_ENABLE_HCA_PEER_AFFINITY": "0",
    "MC_ENABLE_PARALLEL_REG_MR": "1",
    "MC_NIC_PEER_AFFINITY": "",
    "MC_AUTO_GID_MAX_RETRIES": "3",
    "MC_CONN_PAUSE_TTL_MS": "0",
    "MC_CONTEXT_PAUSE_TTL_MS": "0",
    "MC_MAX_CONCURRENT_REG_MR": "16",
    "MC_DISABLE_METACACHE": "0",
    "MC_MAX_PRC_PORT": "0",
    "MC_MAX_RPC_PORT": "0",
    "MC_MIN_PRC_PORT": "0",
    "MC_MIN_RPC_PORT": "0",
    "MC_LEGACY_RPC_PORT_BINDING": "0",
    "MC_HANDSHAKE_CONNECT_TIMEOUT": "300",
    "MC_HANDSHAKE_LISTEN_BACKLOG": "128",
    "MC_HANDSHAKE_MAX_LENGTH": "131072",
    "MC_HANDSHAKE_MAX_NOTIFY_ENTRIES": "32",
    "MC_HANDSHAKE_PORT": "50051",
    "MC_CUSTOM_TOPO_JSON": "",
    "MC_RPC_PROTOCOL": "tcp",
    "MC_TRANSFER_TIMEOUT": "300",
    # ── HIP/KFD (AMD — disabled on Ascend) ──
    "MC_HIP_KFD_PIN_FIX": "0",
    # ── Logging / Diagnostic ──
    "MC_LOG_LEVEL": "WARNING",
    "MC_LOG_DIR": "",
    "MC_LOG_RDMA_SLICE_AFFINITY": "0",
    "MC_USE_TENT": "0",
    "MC_YLT_LOG_LEVEL": "WARNING",
    "MC_TE_METADATA_REFRESH_INTERVAL_SECONDS": "60",
    "MC_TE_METRIC": "0",
    "MC_TE_METRIC_INTERVAL_SECONDS": "60",
    "MC_TE_FILTERS": "",
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
# DeepSeek V4 on Ascend: tp=8, dp=1 (8 NPUs per role)
KV_EXTRA = {
    "mooncake_protocol": "ascend",
    "prefill": {"dp_size": 1, "tp_size": 8},
    "decode":  {"dp_size": 1, "tp_size": 8},
}


def build_kv_config(role: str, rank: int) -> str:
    """Build kv-transfer-config JSON for producer or consumer."""
    # kv_port: P=20001, D=20002 (official convention)
    kv_port = 20001 if role == "kv_producer" else 20002
    cfg = {
        "kv_connector": "MooncakeConnector",
        "kv_buffer_device": "npu",
        "kv_role": role,
        "kv_parallel_size": 1,
        "kv_port": str(kv_port),
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
        "--data-parallel-address", "localhost",
        "--data-parallel-rpc-port", "9100",
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
    return args


def generate_shell_script(name, devices, args_file, logfile):
    """Generate a self-contained bash launch script."""
    lines = ["#!/bin/bash"]
    common_env = [
        "OMP_PROC_BIND=false", "OMP_NUM_THREADS=10",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
    ]
    lines.append("export " + " ".join(common_env))
    # ── Mooncake engine LD_LIBRARY_PATH (system CANN first, mooncake suffix) ──
    MC_LIB_DIR = "/opt/mamba/lib/python3.13/site-packages/mooncake"
    lines.append(f"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{MC_LIB_DIR}")
    # ── ROCE fallback for containers without RDMA NICs ──
    lines.append("export HCCL_INTRA_ROCE_ENABLE=1")
    lines.append("export ASCEND_BUFFER_POOL=4:8")
    mc_vars = " ".join(f"{k}={v}" for k, v in MC_ENV.items())
    if mc_vars:
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
def serve_mode():
    """Read args from _PD_ARGS_FILE and launch vllm API server."""
    import subprocess as sp
    args_file = os.environ.get("_PD_ARGS_FILE", "")
    if not args_file:
        print("[serve] ERROR: _PD_ARGS_FILE not set", file=sys.stderr)
        sys.exit(1)
    with open(args_file) as f:
        cli_args = json.load(f)
    print(f"[serve] Args loaded from {args_file}: {len(cli_args)} tokens")
    cmd = ["vllm", "serve"] + cli_args
    print(f"[serve] Starting: vllm serve ...")
    sys.exit(sp.run(cmd).returncode)


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
        serve_mode()
    elif ns.serial:
        orchestrator_mode()
    else:
        print("Usage: --serial (orchestrator) or set _PD_ARGS_FILE (serve mode)")
        sys.exit(1)


if __name__ == "__main__":
    main()
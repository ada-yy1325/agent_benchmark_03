#!/usr/bin/env python3
"""
PD v3: Single vllm serve with --data-parallel-size 2 via P2PShmConnector.
Installs the connector, applies the AscendStore fix, then execs vllm serve.

Usage (run on remote via tmux):
  python3 perf_opt/scripts/start_pd_v3.py
"""
import os
import shutil
import subprocess
import sys
import time

PROJECT = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test"
MODEL = f"{PROJECT}/models/DeepSeek-V4-Flash-w8a8-npu16"
PKG = "/opt/mamba/lib/python3.13/site-packages"
ASCEND_STORE_DIR = f"{PKG}/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store"
CONNECTOR_FILE = f"{ASCEND_STORE_DIR}/ascend_store_connector.py"
PD_CONNECTOR_FILE = f"{ASCEND_STORE_DIR}/p2p_shm_connector.py"

KV_CONFIG = (
    '{"kv_connector":"P2PShmConnector",'
    '"kv_role":"kv_both","kv_parallel_size":2,'
    '"kv_buffer_size":500000000,'
    '"kv_connector_extra_config":{'
    '"backend":"shm","consumer_is_to_load":true'
    '}}'
)


def apply_fix():
    """Ensure consumer does NOT create LookupKeyServer (DP-aware)."""
    # Delegate to pd_fix_connector.py which handles both v2 and v3
    fix_script = f"{PROJECT}/perf_opt/scripts/pd_fix_connector.py"
    result = subprocess.run([sys.executable, fix_script],
                           capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0:
        print(f"[env] ERROR applying PD fix: {result.stderr}")
        sys.exit(1)


def install_connector():
    """Copy P2PShmConnector to ascend_store and register it."""
    src = f"{PROJECT}/perf_opt/scripts/p2p_shm_connector.py"
    if not os.path.exists(src):
        print(f"[env] ERROR: {src} not found")
        return False
    shutil.copy2(src, PD_CONNECTOR_FILE)
    print(f"[env] Installed P2PShmConnector → {PD_CONNECTOR_FILE}")

    # Patch __init__.py to register the connector during vllm-ascend init
    init_file = f"{PKG}/vllm_ascend/distributed/kv_transfer/__init__.py"
    with open(init_file) as f:
        content = f.read()

    if "P2PShmConnector" not in content:
        # Inject after LMCacheAscendConnector registration
        marker = '        "LMCacheConnectorV1",\n    )'
        injection = (
            '\n'
            '    KVConnectorFactory.register_connector(\n'
            '        "P2PShmConnector",\n'
            '        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.p2p_shm_connector",\n'
            '        "P2PShmConnector",\n'
            '    )\n'
        )
        content = content.replace(marker, marker + injection)
        with open(init_file, "w") as f:
            f.write(content)
        print("[env] Registered P2PShmConnector in __init__.py")
    else:
        print("[env] P2PShmConnector already registered")
    return True


def main():
    print("=" * 60)
    print("PD v3: data-parallel-size=2 + P2PShmConnector")
    print(f"Model: {MODEL}")
    print("=" * 60)

    # Step 1: Cleanup previous runs
    os.system("pkill -9 -f 'VLLM::' 2>/dev/null; sleep 2")
    os.system("fuser -k 8000/tcp 2>/dev/null; fuser -k 8001/tcp 2>/dev/null")
    # NOTE: do NOT delete the torch_npu static kernel cache — it is regenerable
    # but every cold recompile costs minutes; keep it warm. (10-28)
    os.system("ls /root/.cache/torch_npu/static_kernel_* 2>/dev/null")
    os.makedirs("/tmp/pd_test", exist_ok=True)

    # Step 2: Environment setup
    apply_fix()
    if not install_connector():
        sys.exit(1)

    # Step 3: Start vllm serve with --data-parallel-size 2
    cmd = [
        "vllm", "serve", MODEL,
        "--host", "0.0.0.0", "--port", "8000",
        "--max-model-len", "133120",
        "--max-num-batched-tokens", "8192",
        "--served-model-name", "dsv4",
        "--gpu-memory-utilization", "0.9",
        "--max-num-seqs", "64",
        "--data-parallel-size", "2",
        "--tensor-parallel-size", "8",
        "--enable-expert-parallel",
        "--quantization", "ascend",
        "--block-size", "128",
        "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
        "--kv-transfer-config", KV_CONFIG,
        "--additional-config",
        '{"ascend_compilation_config":{'
        '"enable_npugraph_ex":true,"enable_static_kernel":true},'
        '"enable_cpu_binding":true,"enable_dsa_cp":true,'
        '"multistream_overlap_shared_expert":true'
        '}',
    ]

    # Env vars
    env = os.environ.copy()
    env.update({
        "OMP_PROC_BIND": "false",
        "OMP_NUM_THREADS": "10",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "HCCL_BUFFSIZE": "1024",
        "TASK_QUEUE_ENABLE": "1",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
        "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
    })

    logfile = "/tmp/pd_test/pd_v3.log"
    os.makedirs("/tmp/pd_test", exist_ok=True)
    cmd_str = " ".join(cmd)
    print(f"\n[launch] vllm serve → {logfile}")
    print(f"[launch] ENV ASCEND_RT_VISIBLE_DEVICES=0-15")
    print(f"[launch] data-parallel-size=2 | tensor-parallel-size=8")
    print()

    with open(logfile, "w") as log:
        log.write(f"# PD v3: {time.ctime()}\n")
        log.write(f"# CMD: {cmd_str}\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
            cwd=PROJECT,
        )
        print(f"[launch] PID={proc.pid}. Tail log: tail -f {logfile}")
        print(f"[launch] Wait for startup: curl http://127.0.0.1:8000/health")

    # Wait and check health
    time.sleep(60)
    try:
        import urllib.request
        resp = urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5)
        print(f"\n✅ PD v3 healthy! status={resp.status}")
    except Exception as e:
        print(f"\n⏳ Server still starting ({e}). Check: tail -f {logfile}")


if __name__ == "__main__":
    main()
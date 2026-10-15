#!/usr/bin/env python3
"""
Phase 3 topology server launcher.
Accepts --tp (tensor_parallel_size) and --dp (data_parallel_size) overrides.
Usage: python3 start_server_topo.py --tp 16 --dp 1 [--batched N] [--max-seqs N] [--port P]
"""
import os, sys, asyncio, argparse

# ── AOTAutogradCache ──
import torch._functorch.config as _functorch_cfg
_functorch_cfg.enable_autograd_cache = False
_functorch_cfg.bypass_autograd_cache_key = True
print("[launcher] AOTAutogradCache disabled", flush=True)

# ── Env vars ──
os.environ.setdefault("OMP_PROC_BIND", "false")
os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HCCL_BUFFSIZE", "1024")
os.environ.setdefault("TASK_QUEUE_ENABLE", "1")
os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "AIV")
os.environ.setdefault("VLLM_ASCEND_ENABLE_FLASHCOMM1", "1")

# ── Parse overrides ──
parser = argparse.ArgumentParser()
parser.add_argument("--tp", type=int, required=True, help="tensor-parallel-size")
parser.add_argument("--dp", type=int, default=1, help="data-parallel-size")
parser.add_argument("--batched", type=int, default=8192)
parser.add_argument("--max-seqs", type=int, default=64)
parser.add_argument("--port", type=int, default=8000)
args_cmd = parser.parse_args()

TP = args_cmd.tp
DP = args_cmd.dp
BATCHED = args_cmd.batched
MAX_SEQS = args_cmd.max_seqs
PORT = args_cmd.port

WORLD = TP * DP
print(f"[launcher] TP={TP}, DP={DP}, WORLD={WORLD}, batched={BATCHED}, max_seqs={MAX_SEQS}, port={PORT}", flush=True)

# ── Monkey-patch 1: fix single-node data-parallel HCCL rank offset ──
# In vllm v1, MultiprocExecutor computes global_start_rank as:
#   global_start_rank = local_world_size * parallel_config.node_rank_within_dp
# node_rank_within_dp = node_rank % nnodes_within_dp.
# On a single node (nnodes=1), node_rank_within_dp is always 0 for all DP
# groups, so DP1 workers get rank=0..TP-1 (same as DP0), causing HCCL rank
# collision.
# Fix: override node_rank_within_dp to return data_parallel_rank_local for
# single-node multi-DP, so DP1 gets global_start_rank=TP=8.
import vllm.config.parallel as _vllm_parallel
_orig_nrodp = _vllm_parallel.ParallelConfig.node_rank_within_dp

def _new_nrodp(self) -> int:
    if self.nnodes == 1 and self.data_parallel_size > 1:
        # Single-node multi-DP: use data_parallel_rank_local for offset
        return self.data_parallel_rank_local
    return _orig_nrodp.fget(self)

_vllm_parallel.ParallelConfig.node_rank_within_dp = property(_new_nrodp)
print(f"[launcher] Monkey-patched node_rank_within_dp: data_parallel_rank_local for DP", flush=True)

# ── Monkey-patch 2: partition NPU devices via ASCEND_RT_VISIBLE_DEVICES ──
# On NPU, setting ASCEND_RT_VISIBLE_DEVICES per DP group restricts each
# EngineCore+workers to only its own device range, solving device assignment.
# However, the NPU RawDriver always opens physical device 0 during init,
# so we also need a barrier to ensure ALL workers finish set_device before
# ANY worker starts HCCL init (which locks device 0).
# Approach: intercept EngineCoreProc.run_engine_core to set env var + barrier.
import os as _os
import vllm.v1.engine.core as _engine_core
_orig_run_engine_core = _engine_core.EngineCoreProc.run_engine_core

def _patched_run_engine_core(*args, dp_rank=0, local_dp_rank=0, **kwargs):
    vllm_config = kwargs.get("vllm_config")
    if vllm_config is not None:
        pc = vllm_config.parallel_config
        if pc.data_parallel_size > 1:
            tp_size = pc.tensor_parallel_size
            start = dp_rank * tp_size
            end = start + tp_size - 1
            vis_devices = ",".join(str(i) for i in range(start, end + 1))
            _os.environ["ASCEND_RT_VISIBLE_DEVICES"] = vis_devices
            print(f"[DP_PARTITION] dp_rank={dp_rank} "
                  f"ASCEND_RT_VISIBLE_DEVICES={vis_devices}", flush=True)
    return _orig_run_engine_core(*args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs)

_engine_core.EngineCoreProc.run_engine_core = _patched_run_engine_core
print(f"[launcher] Monkey-patched EngineCoreProc.run_engine_core: ASCEND_RT_VISIBLE_DEVICES", flush=True)

# ── Monkey-patch 3: device mapping + barrier before HCCL ──
# Problem 1: NPU device assignment must use global device ID (dp*tp+local)
# Problem 2: NPU RawDriver needs device 0; HCCL locks it. Barrier sync.
import fcntl as _fcntl
import time as _time

_SYNC_FILE = "/tmp/dp2tp8_sync.lock"

# ── Main process: create fresh sync file ──
if __name__ == '__main__':
    try:
        _os.remove(_SYNC_FILE)
    except FileNotFoundError:
        pass
    with open(_SYNC_FILE, "w") as _f:
        _f.write("0\n")
    print(f"[launcher] Created sync file {_SYNC_FILE}", flush=True)

def _sync_barrier(pid):
    """File-based barrier: increment counter, wait until WORLD reached."""
    print(f"[BARRIER] PID={pid} device ready, waiting for all {WORLD} workers...", flush=True)
    done = False
    while not done:
        try:
            with open(_SYNC_FILE, "r+") as _f:
                _fcntl.flock(_f, _fcntl.LOCK_EX)
                content = _f.read().strip()
                count = int(content) if content else 0
                if count < WORLD:
                    count += 1
                    _f.seek(0)
                    _f.write(f"{count}\n")
                    _f.truncate()
                _fcntl.flock(_f, _fcntl.LOCK_UN)
                print(f"[BARRIER] PID={pid} count={count}/{WORLD}", flush=True)
                if count >= WORLD:
                    done = True
        except (ValueError, FileNotFoundError) as e:
            print(f"[BARRIER] PID={pid} sync error: {e}", flush=True)
        if not done:
            _time.sleep(0.5)
    print(f"[BARRIER] PID={pid} proceeding to HCCL init (count={count})", flush=True)

import vllm_ascend.worker.worker as _ascend_worker

# Patch _init_device: map device ID + barrier before HCCL
_orig_init_device = _ascend_worker.NPUWorker._init_device

def _patched_init_device(self):
    saved_local_rank = self.local_rank
    # Compute NPU device: DP group offset + local TP rank
    try:
        dp_rank = self.vllm_config.parallel_config.data_parallel_rank
    except Exception:
        dp_rank = getattr(self.vllm_config.parallel_config, 'data_parallel_rank', 0)
    tp_size = self.vllm_config.parallel_config.tensor_parallel_size
    device_id = dp_rank * tp_size + saved_local_rank
    self.local_rank = device_id
    pid = _os.getpid()
    print(f"[DEV_MAP] PID={pid} dp={dp_rank} tp={tp_size} local={saved_local_rank} → NPU{device_id}", flush=True)
    
    # Call original _init_device with mapped device (set_device + HCCL init inside)
    # BUT: HCCL init locks device 0, so we need to sync before that.
    # Actually, _init_device calls _init_worker_distributed_environment (HCCL).
    # We'll patch that method separately.
    try:
        return _orig_init_device(self)
    finally:
        self.local_rank = saved_local_rank

_ascend_worker.NPUWorker._init_device = _patched_init_device
print(f"[launcher] Monkey-patched _init_device: device=dp*tp+local", flush=True)

# Patch _init_worker_distributed_environment for barrier before HCCL
_orig_init_distributed = _ascend_worker.NPUWorker._init_worker_distributed_environment

def _patched_init_distributed(self):
    pid = _os.getpid()
    _sync_barrier(pid)
    return _orig_init_distributed(self)

_ascend_worker.NPUWorker._init_worker_distributed_environment = _patched_init_distributed
print(f"[launcher] Monkey-patched _init_worker_distributed_environment: barrier before HCCL", flush=True)

MODEL_DIR = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-self"

CLI_ARGS = [
    MODEL_DIR,
    '--host', '0.0.0.0',
    '--port', str(PORT),
    '--max-model-len', '133120',
    '--max-num-batched-tokens', str(BATCHED),
    '--served-model-name', 'dsv4',
    '--gpu-memory-utilization', '0.9',
    '--max-num-seqs', str(MAX_SEQS),
    '--data-parallel-size', str(DP),
    '--tensor-parallel-size', str(TP),
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

if __name__ == "__main__":
    print(f"[launcher] {' '.join(CLI_ARGS)}", flush=True)

    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args

    p = FlexibleArgumentParser(description="vLLM API server (topo)")
    p = make_arg_parser(p)
    a = p.parse_args(CLI_ARGS)
    validate_parsed_serve_args(a)

    if hasattr(a, 'model_tag') and a.model_tag is not None:
        a.model = a.model_tag
    if hasattr(a, 'tokenizer_tag') and a.tokenizer_tag is not None:
        a.tokenizer = a.tokenizer_tag

    from vllm.entrypoints.openai.api_server import run_server
    asyncio.run(run_server(a))
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
# node_rank_within_dp = node_rank % nnodes_within_dp, and nnodes_within_dp
# returns 1 when nnodes==1, so both DP groups get global_start_rank=0.
# Fix: set nnodes=2, node_rank=dp_rank so node_rank_within_dp=dp_rank,
# giving DP1 global_start_rank=TP=8 (workers get rank 8-15 for HCCL).
import vllm.v1.engine.core as _engine_core
_orig_run_engine_core = _engine_core.EngineCoreProc.run_engine_core

def _patched_run_engine_core(*args, dp_rank=0, local_dp_rank=0, **kwargs):
    vllm_config = kwargs.get("vllm_config")
    if vllm_config is not None and dp_rank > 0:
        pc = vllm_config.parallel_config
        if pc.data_parallel_size > 1:
            old_nnodes = pc.nnodes
            old_node_rank = pc.node_rank
            old_dp_size_local = pc.data_parallel_size_local
            pc.nnodes = max(pc.nnodes, pc.data_parallel_size)
            pc.node_rank = dp_rank
            pc.data_parallel_size_local = pc.data_parallel_size
            print(f"[DP_FIX] dp_rank={dp_rank} "
                  f"nnodes {old_nnodes}→{pc.nnodes} "
                  f"node_rank {old_node_rank}→{pc.node_rank} "
                  f"dp_size_local {old_dp_size_local}→{pc.data_parallel_size_local}", flush=True)
    return _orig_run_engine_core(*args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs)

_engine_core.EngineCoreProc.run_engine_core = _patched_run_engine_core
print(f"[launcher] Monkey-patched EngineCoreProc.run_engine_core: nnodes+node_rank for DP", flush=True)

# ── Monkey-patch 2: fix single-node data-parallel NPU device assignment ──
# In vllm-ascend, NPUWorker._init_device selects device via self.local_rank (0..TP-1).
# On a single node with DP>1, both DP groups get local_rank 0..TP-1, causing
# DP1 workers to try opening already-used NPU 0..TP-1 → SetDevice error 507033.
# Even with fixed HCCL ranks (patch 1), workers still use local_rank for set_device.
# Fix: compute device ID from data_parallel_rank * tp_size + local_rank.
# e.g. DP0→NPU 0-7, DP1→NPU 8-15 without relying on global_start_rank or rank.
import vllm_ascend.worker.worker as _ascend_worker
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
    import os as _os
    print(f"[PATCH] PID={_os.getpid()} "
          f"dp_rank={dp_rank} tp_size={tp_size} local={saved_local_rank} "
          f"→ NPU{device_id}", flush=True)
    try:
        return _orig_init_device(self)
    finally:
        self.local_rank = saved_local_rank

_ascend_worker.NPUWorker._init_device = _patched_init_device
print(f"[launcher] Monkey-patched NPUWorker._init_device: device=dp_rank*tp+local", flush=True)

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
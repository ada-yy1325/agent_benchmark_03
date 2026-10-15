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

# ── Monkey-patch 1: fix single-node data-parallel DP rank offset ──
# In vllm v1, MultiprocExecutor computes global_start_rank as:
#   global_start_rank = local_world_size * parallel_config.node_rank_within_dp
# On a single node with DP>1, node_rank_within_dp defaults to 0 for all DP groups,
# so both DP groups assign global rank 0..TP-1 to workers. This causes:
#   1. All workers mapped to NPU 0..TP-1 (device conflict)
#   2. HCCL rank collision (16 workers sharing 8 unique ranks)
# Fix: set node_rank_within_dp = data_parallel_rank, so DP1 gets global_start_rank=TP
import vllm.v1.engine.core as _engine_core
_orig_run_engine_core = _engine_core.EngineCoreProc.run_engine_core

def _patched_run_engine_core(*args, dp_rank=0, local_dp_rank=0, **kwargs):
    vllm_config = kwargs.get("vllm_config")
    if vllm_config is not None:
        parallel_config = vllm_config.parallel_config
        if parallel_config.data_parallel_size > 1:
            # Single-node multi-DP: map each DP group to its own device range
            parallel_config.node_rank_within_dp = dp_rank
            print(f"[DP_FIX] dp_rank={dp_rank} node_rank_within_dp={dp_rank} "
                  f"local_world_size={parallel_config.tensor_parallel_size} "
                  f"→ global_start_rank={parallel_config.tensor_parallel_size * dp_rank}",
                  flush=True)
    return _orig_run_engine_core(*args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs)

_engine_core.EngineCoreProc.run_engine_core = _patched_run_engine_core
print(f"[launcher] Monkey-patched EngineCoreProc.run_engine_core: node_rank_within_dp=dp_rank", flush=True)

# ── Monkey-patch 2: fix single-node data-parallel NPU device assignment ──
# In vllm-ascend, NPUWorker._init_device selects device via self.local_rank.
# Even after fixing global_start_rank (patch 1), worker NPUWorker still sees
# local_rank=0..TP-1 (correct within-group rank, but wrong for device selection
# when multiple DP groups share one node).
# Fix: use self.rank (global rank) instead of self.local_rank for NPU device selection,
# so DP1 workers use npu:TP..npu:(TP+TP-1) correctly.
import vllm_ascend.worker.worker as _ascend_worker
_orig_init_device = _ascend_worker.NPUWorker._init_device

def _patched_init_device(self):
    saved_local_rank = self.local_rank
    saved_rank = self.rank
    self.local_rank = self.rank  # Use global rank for NPU device selection
    import os as _os
    print(f"[PATCH] PID={_os.getpid()} local_rank({saved_local_rank}) → rank({saved_rank}) "
          f"→ NPU{self.local_rank}", flush=True)
    try:
        return _orig_init_device(self)
    finally:
        self.local_rank = saved_local_rank

_ascend_worker.NPUWorker._init_device = _patched_init_device
print(f"[launcher] Monkey-patched NPUWorker._init_device: device=rank (not local_rank)", flush=True)

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
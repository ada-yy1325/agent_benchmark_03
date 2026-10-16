#!/usr/bin/env python3
"""Probe: what EP configs does vLLM-Ascend support on 16 NPU?"""
import os, sys

os.environ.setdefault("OMP_PROC_BIND", "false")
os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HCCL_BUFFSIZE", "1024")
os.environ.setdefault("TASK_QUEUE_ENABLE", "1")
os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "AIV")
os.environ.setdefault("VLLM_ASCEND_ENABLE_FLASHCOMM1", "1")

# Print critical EP-related source
import vllm.config.parallel as vp
import inspect

print("=== ParallelConfig class ===")
# Show expert_parallel_size property
src = inspect.getsource(vp.ParallelConfig)
# Find expert_parallel_size
for line in src.split('\n'):
    if 'expert' in line.lower() or 'parallel_size' in line.lower() or 'world' in line.lower():
        print(f"  {line.strip()}")

print()
print("=== Enable expert parallel flag ===")
# Check CLI args
import vllm.engine.arg_utils as arg_utils
src2 = inspect.getsource(arg_utils)
for line in src2.split('\n'):
    if 'expert' in line.lower() and ('parallel' in line.lower() or 'enable' in line.lower()):
        print(f"  {line.strip()}")

print()
print("=== vLLM version ===")
import vllm
print(f"  vllm version: {vllm.__version__}")
print(f"  vllm commit: {getattr(vllm, '__commit__', 'N/A')}")

print()
print("=== Test config parsing ===")
from vllm.config import ParallelConfig
# Try tp=8
pc = ParallelConfig(tensor_parallel_size=8, pipeline_parallel_size=1)
print(f"  tp=8, dp=1: world={pc.world_size}, ep={pc.expert_parallel_size}")

pc = ParallelConfig(tensor_parallel_size=8, data_parallel_size=2, pipeline_parallel_size=1)
print(f"  tp=8, dp=2: world={pc.world_size}, ep={pc.expert_parallel_size}")

pc = ParallelConfig(tensor_parallel_size=16, pipeline_parallel_size=1)
print(f"  tp=16, dp=1: world={pc.world_size}, ep={pc.expert_parallel_size}")

# Can we set expert_parallel_size explicitly?
try:
    pc = ParallelConfig(tensor_parallel_size=8, enable_expert_parallel=True)
    print(f"  tp=8, enable_ep=True: world={pc.world_size}, ep={pc.expert_parallel_size}")
except Exception as e:
    print(f"  tp=8, enable_ep=True: ERROR {e}")
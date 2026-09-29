#!/usr/bin/env python3
"""Try to load DeepSeek V4 Flash on Ascend 910B."""
import argparse, json, os, sys, time

def check_download():
    """Check download status."""
    d = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash"
    if not os.path.exists(d):
        print("Model directory not found!")
        return False
    idx = os.path.join(d, "model.safetensors.index.json")
    if not os.path.exists(idx):
        print("Index file not yet downloaded...")
        return False
    with open(idx) as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})
    total_shards = len(set(weight_map.values()))
    complete = [s for s in set(weight_map.values()) if os.path.exists(os.path.join(d, s))]
    print(f"Shards: {len(complete)}/{total_shards} complete")
    return len(complete) == total_shards

def dummy_load():
    """Load with dummy weights to test framework path."""
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # 禁用 UVA offloader（NPU 上不支持），回退到 functional_call 方案
    import os
    os.environ["VLLM_WEIGHT_OFFLOADING_DISABLE_UVA"] = "1"

    model_path = "./models/DeepSeek-V4-Flash"
    print(f"Model: {model_path} (dummy weights)")
    import torch
    print(f"NPU: {torch.npu.device_count()} devices")
    for i in range(torch.npu.device_count()):
        props = torch.npu.get_device_properties(i)
        print(f"  [{i}] {props.name} - {props.total_memory / 1024**3:.1f}GB")
    print()

    from vllm import LLM, SamplingParams

    config = {
        "model": model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 512,
        "enforce_eager": True,
        "dtype": "bfloat16",
        "max_num_seqs": 1,
        "load_format": "dummy",
    }

    print("Creating LLM with dummy config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    llm = LLM(**config)
    print("Dummy model loaded successfully!")

    prompt = "Hello, what is the capital of France?"
    sp = SamplingParams(temperature=0, max_tokens=20)
    outputs = llm.generate([prompt], sp)
    for o in outputs:
        print(f"Prompt: {o.prompt}")
        print(f"Output: {o.outputs[0].text}")

    return llm

def try_load():
    """Try to load the model with real weights."""
    # NPU 需要 spawn 而不是 fork 来初始化多进程
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    # 禁用 UVA offloader（NPU 上不支持），回退到 functional_call 方案
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    import os
    os.environ["VLLM_WEIGHT_OFFLOADING_DISABLE_UVA"] = "1"

    model_path = "./models/DeepSeek-V4-Flash"
    print(f"Model: {model_path}")
    import torch
    print(f"NPU: {torch.npu.device_count()} devices")
    for i in range(torch.npu.device_count()):
        props = torch.npu.get_device_properties(i)
        print(f"  [{i}] {props.name} - {props.total_memory / 1024**3:.1f}GB")
    print()
    
    from vllm import LLM, SamplingParams
    
    config = {
        "model": model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 512,
\"max_model_len\": 512,
        "enforce_eager": True,
        # 不传 quantization，让 DeepseekV4FP8Config.override_quantization_method
        # 自动检测 model_type=="deepseek_v4" 并选择 deepseek_v4_fp8
        # cpu_offload_gb 在 NPU 上需设置 VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1
        # 以禁用 UVA 路径，回退到 functional_call 方案
        "dtype": "bfloat16",
        "max_num_seqs": 1,
    }
    
    print("Creating LLM with config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    llm = LLM(**config)
    print("Model loaded successfully!")
    
    prompt = "Hello, what is the capital of France?"
    sp = SamplingParams(temperature=0, max_tokens=20)
    outputs = llm.generate([prompt], sp)
    for o in outputs:
        print(f"Prompt: {o.prompt}")
        print(f"Output: {o.outputs[0].text}")
    
    return llm

if __name__ == "__main__":
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Just check download status")
    parser.add_argument("--dummy", action="store_true", help="Load with dummy weights")
    args = parser.parse_args()
    
    if args.dry_run:
        check_download()
    elif args.dummy:
        dummy_load()
    else:
        try_load()
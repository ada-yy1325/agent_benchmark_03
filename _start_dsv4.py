#!/usr/bin/env python3
"""Try to load DeepSeek V4 Flash on Ascend 910B."""
import argparse, json, os, sys, time
# 必须在任何 import 之前设置 HCCL 环境变量（4 张 NPU 跨两个模块，需用 NIC 通信）
os.environ["HCCL_OVER_NIC"] = "1"
os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"
os.environ["HCCL_CONNECT_TIMEOUT"] = "3600"

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
    import os
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # 禁用 UVA offloader（NPU 上不支持），回退到 functional_call 方案
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
        "moe_backend": "vanilla",
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
    import os
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["HCCL_OVER_NIC"] = "1"
    os.environ["HCCL_INTRA_ROCE_ENABLE"] = "0"
    os.environ["HCCL_P2P_DISABLE"] = "1"
    # 禁用 UVA offloader（NPU 上不支持），回退到 functional_call 方案
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
    # Monkey-patch: skip Float4 format_cast unsupported on 910B2C
    try:
        import vllm_ascend.quantization.methods.fp8 as _fp8_module
        _orig_process = _fp8_module.AscendW4A8MXFPDSDynamicFusedMoEMethod.process_weights_after_loading

        def _patched_process(self, layer):
            try:
                _orig_process(self, layer)
            except RuntimeError as e:
                err_str = str(e)
                if any(kw in err_str.lower() for kw in ["customize_dtype", "not supported", "float4"]):
                    print(f"[WARN] Skipping Float4 format_cast (910B2C unsupported): {err_str[:80]}")
                else:
                    raise

        _fp8_module.AscendW4A8MXFPDSDynamicFusedMoEMethod.process_weights_after_loading = _patched_process
        print("[INFO] Applied monkey-patch for 910B2C Float4 format_cast")
    except (ImportError, AttributeError) as e:
        print(f"[INFO] No patch needed: {e}")
    
    config = {
        "model": model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 512,
        "enforce_eager": True,
        # 不传 quantization，让 DeepseekV4FP8Config.override_quantization_method
        # 自动检测 model_type=="deepseek_v4" 并选择 deepseek_v4_fp8
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
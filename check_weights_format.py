"""检查 W8A8 和 W8A16 的权重格式差异"""
import json
import safetensors
from safetensors import safe_open

base = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models"

# 检查 W8A8
print("=" * 60)
print("W8A8 模型权重格式")
print("=" * 60)
with safe_open(f"{base}/Qwen3-4B-Base-W8A8/model-00001-of-00002.safetensors", framework="pt") as f:
    keys = list(f.keys())
    print(f"Total tensors: {len(keys)}")
    layer0_keys = [k for k in keys if "layers.0.self_attn.q_proj" in k][:5]
    for k in layer0_keys:
        tensor = f.get_tensor(k)
        print(f"  {k}: shape={list(tensor.shape)}, dtype={tensor.dtype}")

# 检查 W8A16
print()
print("=" * 60)
print("W8A16 模型权重格式")
print("=" * 60)
with safe_open(f"{base}/Qwen3-4B-Base-W8A16/model-00001-of-00001.safetensors", framework="pt") as f:
    keys = list(f.keys())
    print(f"Total tensors: {len(keys)}")
    layer0_keys = [k for k in keys if "layers.0.self_attn.q_proj" in k][:5]
    for k in layer0_keys:
        tensor = f.get_tensor(k)
        print(f"  {k}: shape={list(tensor.shape)}, dtype={tensor.dtype}")

# 对比差异
print()
print("=" * 60)
print("完整 tensor 列表对比 (q_proj)")
print("=" * 60)
for model_name in ["Qwen3-4B-Base-W8A8", "Qwen3-4B-Base-W8A16"]:
    fname = f"{base}/{model_name}"
    import glob
    import os
    files = sorted(glob.glob(f"{fname}/*.safetensors"))
    print(f"\n--- {model_name} ---")
    with safe_open(files[0], framework="pt") as f:
        qkeys = [k for k in f.keys() if "layers.0.self_attn.q_proj" in k]
        for k in sorted(qkeys):
            t = f.get_tensor(k)
            print(f"  {k}: {list(t.shape)} {t.dtype}  min={t.min().item():.4f}  max={t.max().item():.4f}")
"""检查 DeepSeek V4 模型权重文件的存储格式 - 重点找专家权重"""
import os
import json

model_dir = "./models/DeepSeek-V4-Flash"
files = sorted([f for f in os.listdir(model_dir) if f.endswith(".safetensors")])
print(f"Total safetensors files: {len(files)}")

# 找专家权重和MoE gate的dtype
found_experts = False
total_expert_params = 0
total_params = 0
dtype_count = {}

for fname in files:
    fpath = os.path.join(model_dir, fname)
    with open(fpath, "rb") as f:
        length_bytes = f.read(8)
        metadata_len = int.from_bytes(length_bytes, "little")
        f.seek(8)
        metadata_json = f.read(metadata_len).decode("utf-8")
        metadata = json.loads(metadata_json)
        
    tensors = [k for k in metadata if k != "__metadata__"]
    
    for t in tensors:
        info = metadata[t]
        shape = info.get("shape", [])
        dtype = info.get("dtype", "?")
        dtype_count[dtype] = dtype_count.get(dtype, 0) + 1
        
        # 计算参数量
        param_count = 1
        for s in shape:
            param_count *= s
        total_params += param_count
        
        # 查找专家权重
        if any(p in t for p in ["down_proj", "gate_proj", "up_proj"]):
            total_expert_params += param_count
            if not found_experts:
                print(f"Found expert weight: {t}")
                print(f"  dtype: {dtype}, shape: {shape}")
                found_experts = True

print(f"\n=== Dtype distribution ===")
for dt, cnt in sorted(dtype_count.items()):
    print(f"  {dt}: {cnt} tensors")

print(f"\n=== Parameter count ===")
print(f"Total parameters (from shapes): {total_params / 1e9:.2f}B")
print(f"Expert params (gate/up/down): {total_expert_params / 1e9:.2f}B")
print(f"Non-expert params: {(total_params - total_expert_params) / 1e9:.2f}B")

# 内存估算
print(f"\n=== Memory estimation ===")
print(f"BF16 (2B/param): {total_params * 2 / 1024**3:.1f} GB")
print(f"Mixed FP8+FP4 (experts=FP4, rest=FP8): {(total_expert_params * 0.5 + (total_params - total_expert_params) * 1) / 1024**3:.1f} GB")
print(f"Disk size: {sum(os.path.getsize(os.path.join(model_dir, f)) for f in files) / 1024**3:.2f} GB")
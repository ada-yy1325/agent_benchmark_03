"""检查 DeepSeek V4 模型权重文件的存储格式"""
import os
import json

model_dir = "./models/DeepSeek-V4-Flash"
files = sorted([f for f in os.listdir(model_dir) if f.endswith(".safetensors")])
print(f"Total safetensors files: {len(files)}")

# 检查几个文件的元数据
for fname in files[:3]:
    fpath = os.path.join(model_dir, fname)
    with open(fpath, "rb") as f:
        length_bytes = f.read(8)
        metadata_len = int.from_bytes(length_bytes, "little")
        f.seek(8)
        metadata_json = f.read(metadata_len).decode("utf-8")
        metadata = json.loads(metadata_json)
        print(f"\n=== {fname} ===")
        tensors = [k for k in metadata if k != "__metadata__"]
        print(f"Total tensors: {len(tensors)}")
        # 显示前 5 个
        for t in tensors[:5]:
            info = metadata[t]
            print(f"  {t}: dtype={info.get('dtype','?')}, shape={info.get('shape','?')}")
        # 看看后面几个专家权重
        expert_tensors = [t for t in tensors if "experts" in t and "gate_proj" in t]
        if expert_tensors:
            e = expert_tensors[0]
            print(f"  Sample expert gate: {e}: dtype={metadata[e].get('dtype','?')}, shape={metadata[e].get('shape','?')}")
        print(f"  Metadata: {json.dumps(metadata.get('__metadata__',{}), indent=2)[:200]}")

# 文件总大小
total_size = sum(os.path.getsize(os.path.join(model_dir, f)) for f in files)
print(f"\nTotal size on disk: {total_size/1024**3:.2f} GB")
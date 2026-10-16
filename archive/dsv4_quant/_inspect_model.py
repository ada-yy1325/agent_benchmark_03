"""检查 DeepSeek V4 模型权重文件结构 - 找 expert 权重命名"""
# 补一个：看看专家权重实际用了什么格式
# 前一层推断 F8_E4M3 是块级量化的权重

# 看看 layers.0.ffn.experts 的结构
expert_keys = [t for t in tensors if "experts" in t]
w1_weights = [t for t in expert_keys if "w1" in t and "scale" not in t]
if w1_weights:
    t = w1_weights[0]
    print(f"\nExpert weight sample: {t}")
    print(f"  dtype: {metadata[t].get('dtype')}, shape: {metadata[t].get('shape')}")
w1_scales = [t for t in expert_keys if "w1" in t and "scale" in t]
if w1_scales:
    t = w1_scales[0]
    print(f"Expert scale sample: {t}")
    print(f"  dtype: {metadata[t].get('dtype')}, shape: {metadata[t].get('shape')}")
import os
import json

model_dir = "./models/DeepSeek-V4-Flash"
files = sorted([f for f in os.listdir(model_dir) if f.endswith(".safetensors")])
print(f"Total files: {len(files)}")

# 看第2个文件的全部tensor名字（包含各层权重）
fname = files[1]
fpath = os.path.join(model_dir, fname)
with open(fpath, "rb") as f:
    length_bytes = f.read(8)
    metadata_len = int.from_bytes(length_bytes, "little")
    f.seek(8)
    metadata_json = f.read(metadata_len).decode("utf-8")
    metadata = json.loads(metadata_json)

tensors = [k for k in metadata if k != "__metadata__"]
print(f"\n=== All 4-level prefixes in {fname} ===")
by_prefix = {}
for t in tensors:
    parts = t.split(".")
    prefix = ".".join(parts[:4]) if len(parts) >= 4 else t
    if prefix not in by_prefix:
        by_prefix[prefix] = []
    by_prefix[prefix].append(t)

for pref in sorted(by_prefix.keys()):
    names = by_prefix[pref]
    sample_names = names[:3]
    infos = [f"{n}: {metadata[n].get('dtype','?')} {metadata[n].get('shape','?')}" for n in sample_names]
    print(f"\n  {pref} ({len(names)} tensors):")
    for s in infos:
        print(f"    {s}")
    if len(names) > 3:
        print(f"    ... and {len(names)-3} more")
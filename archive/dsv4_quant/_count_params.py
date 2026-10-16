"""Count total parameters and memory requirements for GLM-5.3-Flash."""
import json, os, glob
from safetensors import safe_open

model_dir = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/GLM-5.3-Flash"

# Check index file
index_file = os.path.join(model_dir, "model.safetensors.index.json")
if os.path.exists(index_file):
    with open(index_file) as f:
        idx = json.load(f)
    print("=== model.safetensors.index.json ===")
    metadata = idx.get("metadata", {})
    print(f"Total params from metadata: {metadata.get('total_params', 'N/A')}")
    weight_map = idx.get("weight_map", {})
    print(f"Total tensors in weight map: {len(weight_map)}")
    safetensor_files = set(weight_map.values())
    print(f"Number of safetensor files: {len(safetensor_files)}")

# Scan all safetensor files to count parameters
files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
print(f"\nTotal safetensor files: {len(files)}")
print(f"Total disk size: ~306 GB")

# Count elements from all files
total_elements = 0
total_bf16_bytes = 0
file_count = 0
for fname in files:
    file_count += 1
    try:
        with safe_open(fname, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                total_elements += tensor.numel()
                # element_size() returns bytes per element
                total_bf16_bytes += tensor.numel() * 2  # 2 bytes for BF16
    except Exception as e:
        print(f"  Error reading {os.path.basename(fname)}: {e}")
    if file_count % 10 == 0:
        print(f"  Scanned {file_count}/{len(files)} files... ({total_elements/1e9:.1f}B elements so far)")

print(f"\n{'='*60}")
print(f"Total elements: {total_elements/1e9:.2f}B")
print(f"Memory at FP8  (1 byte/elem): {total_elements/1e9:.1f} GB")
print(f"Memory at BF16 (2 bytes/elem): {total_elements*2/1e9:.1f} GB")
print(f"Memory at INT8 (1 byte/elem): {total_elements/1e9:.1f} GB")
print(f"{'='*60}")
print(f"HBM available: 4 × 64 GB = 256 GB")
print(f"CPU RAM available for offload: ~120 GB")
"""Load GLM-5.3-Flash with explicit per-device memory limits on 16 NPUs."""
import os, torch, json

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"

import torch_npu
print(f"torch_npu version: {torch_npu.__version__}")
print(f"NPU device count: {torch_npu.npu.device_count()}")

model_path = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/GLM-5.3-Flash"

from transformers.models.glm5_next import Glm5NextForConditionalGeneration
from transformers import AutoTokenizer

print(f"Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print(f"Tokenizer: {type(tokenizer).__name__}")

# Set explicit max_memory per device (leave headroom for activations)
# 16 NPUs × ~58 GB usable (leave ~6 GB per card for activations/init)
max_memory = {i: "58GiB" for i in range(16)}
max_memory["cpu"] = "200GiB"  # Reserve CPU memory for offload

print(f"\nLoading model with per-device memory limits...")
print(f"max_memory: {max_memory}")

model = Glm5NextForConditionalGeneration.from_pretrained(
    model_path,
    device_map="auto",
    max_memory=max_memory,
    torch_dtype="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
print(f"Model loaded: {type(model).__name__}")

# Show device distribution
if hasattr(model, 'hf_device_map') and model.hf_device_map:
    dev_counts = {}
    for mod, dev in model.hf_device_map.items():
        dev_counts[dev] = dev_counts.get(dev, 0) + 1
    for dev, count in sorted(dev_counts.items(), key=lambda x: str(x[0])):
        print(f"  Device {dev}: {count} modules")
else:
    print("No hf_device_map")

# Memory usage per device
print(f"\nMemory usage after loading:")
for i in range(16):
    try:
        mem = torch_npu.npu.memory_allocated(i) / 1e9
        total = torch_npu.npu.get_device_properties(i).total_memory / 1e9
        print(f"  NPU {i}: {mem:.1f} GB / {total:.1f} GB")
    except:
        pass

# Simple test
print(f"\nTesting generation...")
prompt = "Hello, what is 1+1?"
inputs = tokenizer(prompt, return_tensors="pt")

# Move inputs to NPU 0
inputs = {k: v.to("npu:0") for k, v in inputs.items()}

outputs = model.generate(**inputs, max_new_tokens=20)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Response: {response}")
print("\n=== SUCCESS ===")
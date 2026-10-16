"""Load GLM-5.3-Flash FP8 with CPU dequant then NPU dispatch."""
import os, torch, gc
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
import torch_npu

model_path = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/GLM-5.3-Flash"
n_npus = torch_npu.npu.device_count()
print(f"NPUs: {n_npus}")

from transformers import AutoTokenizer
from transformers.models.glm5_next import Glm5NextForConditionalGeneration
from accelerate import dispatch_model, infer_auto_device_map

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print(f"Tokenizer: {type(tokenizer).__name__}")

# Step 1: Load model on CPU (FP8 dequant happens on CPU since no device_map)
print(f"Loading model on CPU for FP8->BF16 dequant...")
model = Glm5NextForConditionalGeneration.from_pretrained(
    model_path,
    device_map=None,  # Stay on CPU for FP8 dequant
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
print(f"Model loaded on CPU: {type(model).__name__}")

# Step 2: Check memory on model
param_count = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {param_count/1e9:.1f}B")
mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
print(f"Model size in memory: {mem_mb:.1f} GB")

# Step 3: Dispatch to NPUs
print(f"\nDispatching to {n_npus} NPUs...")
# Create device map - distribute layers evenly
from transformers.models.glm5_next import Glm5NextConfig
config = Glm5NextConfig.from_pretrained(model_path)
n_layers = config.num_hidden_layers  # 45

device_map = {
    "model.embed_tokens": 0,
    "model.visual": 0,
}
layers_per_device = (n_layers + n_npus - 2) // (n_npus - 1)  # Spread across NPUs 0-14, keep 15 for final
for i in range(n_layers):
    dev = min(i // layers_per_device, n_npus - 2) if n_npus > 1 else 0
    device_map[f"model.layers.{i}"] = dev

device_map["model.norm"] = n_npus - 1
device_map["model.final_layernorm"] = n_npus - 1 if hasattr(config, 'final_layernorm') else n_npus - 1
device_map["lm_head"] = n_npus - 1

print(f"Device map ({len(device_map)} entries):")
dev_counts = {}
for k, v in device_map.items():
    dev_counts[v] = dev_counts.get(v, 0) + 1
for d in sorted(dev_counts.keys()):
    print(f"  NPU {d}: {dev_counts[d]} modules")

model = dispatch_model(model, device_map=device_map)
print(f"Model dispatched!")

# Memory per device
for i in range(n_npus):
    try:
        mem = torch_npu.npu.memory_allocated(i) / 1e9
        total = torch_npu.npu.get_device_properties(i).total_memory / 1e9
        print(f"  NPU{i}: {mem:.1f}/{total:.1f} GB")
    except: pass

# Generate
prompt = "Hello!"
inputs = tokenizer(prompt, return_tensors="pt").to("npu:0")
out = model.generate(**inputs, max_new_tokens=10)
print(f"Response: {tokenizer.decode(out[0])}")
print("=== SUCCESS ===")
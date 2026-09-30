"""Quick test: load GLM-5.3-Flash FP8 and run inference."""
import os, torch
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
import torch_npu

model_path = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/GLM-5.3-Flash"
n_npus = torch_npu.npu.device_count()
print(f"NPUs: {n_npus}")

from transformers import AutoTokenizer
from transformers.models.glm5_next import Glm5NextForConditionalGeneration

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print(f"Tokenizer: {type(tokenizer).__name__}")

# Load model - let Transformers handle FP8 dequant
print(f"Loading model (let Transformers handle FP8->BF16)...")
model = Glm5NextForConditionalGeneration.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
print(f"Loaded: {type(model).__name__}")

# Check device distribution
if hasattr(model, 'hf_device_map'):
    devs = {}
    for m, d in model.hf_device_map.items():
        devs[d] = devs.get(d, 0) + 1
    for d, c in sorted(devs.items(), key=lambda x: str(x[0])):
        print(f"  {d}: {c} modules")

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
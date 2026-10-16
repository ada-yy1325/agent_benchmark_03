"""Load GLM-5.3-Flash with Transformers on 16 NPUs - direct class import."""
import os, torch, json

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"

import torch_npu
print(f"torch_npu version: {torch_npu.__version__}")
print(f"NPU device count: {torch_npu.npu.device_count()}")
for i in range(torch_npu.npu.device_count()):
    print(f"  NPU {i}: {torch_npu.npu.get_device_name(i)} - {torch_npu.npu.get_device_properties(i).total_memory / 1e9:.1f} GB")

model_path = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/GLM-5.3-Flash"

# Direct import instead of AutoModel
from transformers.models.glm5_next import Glm5NextForConditionalGeneration
from transformers import AutoTokenizer

print(f"Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print(f"Tokenizer: {type(tokenizer).__name__}")

print(f"\nLoading model with from_pretrained (device_map='auto')...")
print(f"This will dequantize FP8 to BF16 (~643 GB), fitting in 1024 GB HBM...")

model = Glm5NextForConditionalGeneration.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
print(f"Model loaded: {type(model).__name__}")

# Show device map
if hasattr(model, 'hf_device_map') and model.hf_device_map:
    devices = set(model.hf_device_map.values())
    print(f"Devices used: {devices}")
    print(f"Number of modules: {len(model.hf_device_map)}")
else:
    print("No hf_device_map (single device?)")

# Simple generation test
print(f"\nTesting generation...")
prompt = "Hello, what is 1+1?"
inputs = tokenizer(prompt, return_tensors="pt")

# Move to model device (if single device) or first NPU
if hasattr(model, 'device') and str(model.device) != 'cpu':
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
elif torch_npu.npu.device_count() > 0:
    inputs = {k: v.to(f"npu:0") for k, v in inputs.items()}

outputs = model.generate(**inputs, max_new_tokens=20)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Response: {response}")
print("\n=== SUCCESS ===")
"""Load GLM-5.3-Flash with Transformers on 16 NPUs - use AutoModel not AutoModelForCausalLM."""
import os, torch, json

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"

# Import torch_npu first to register NPU devices
import torch_npu
print(f"torch_npu version: {torch_npu.__version__}")
print(f"NPU device count: {torch_npu.npu.device_count()}")
print(f"NPU device name: {torch_npu.npu.get_device_name(0) if torch_npu.npu.device_count() > 0 else 'none'}")
print(f"CUDA device count (after npu import): {torch.cuda.device_count()}")

model_path = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/GLM-5.3-Flash"

from transformers import AutoModel, AutoTokenizer

print(f"\nLoading tokenizer from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print(f"Tokenizer loaded: {type(tokenizer).__name__}")

print("\nLoading model with AutoModel...")
model = AutoModel.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
print(f"Model loaded: {type(model).__name__}")
print(f"Model device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")

# Try a simple generation
print("\nTesting generation...")
prompt = "Hello, what is 1+1?"
inputs = tokenizer(prompt, return_tensors="pt")
# Move inputs to the same device as the model
if hasattr(model, 'device'):
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

outputs = model.generate(**inputs, max_new_tokens=20)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Response: {response}")
print("\n=== SUCCESS ===")
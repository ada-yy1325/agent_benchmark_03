"""Load GLM-5.3-Flash with Transformers on 16 NPUs using device_map='auto'.
FP8 dequantization to BF16 (~643 GB) should fit in 1024 GB HBM."""
import os, torch, json

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch_npu

model_path = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/GLM-5.3-Flash"

print(f"Loading tokenizer from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print(f"Tokenizer loaded: {type(tokenizer).__name__}")

print("Loading model with device_map='auto'...")
print(f"Available NPU count: {torch.cuda.device_count()}")
print(f"NPU device name: {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'none'}")

model = AutoModelForCausalLM.from_pretrained(
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
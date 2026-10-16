"""Check model vs checkpoint key mismatch."""
import torch, os, json
from transformers.models.glm5_next import Glm5NextConfig
import transformers.models.glm5_next as glm5_module

model_path = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/GLM-5.3-Flash"

# 1. Check model parameter names
config = Glm5NextConfig.from_pretrained(model_path)
print(f"Config model_type: {config.model_type}")

with torch.device("meta"):
    model = glm5_module.Glm5NextForConditionalGeneration(config)

model_params = list(model.state_dict().keys())
print(f"\nModel has {len(model_params)} parameters")
print(f"Sample params:")
for p in model_params[:10]:
    print(f"  {p}")

# 2. Check checkpoint keys
from safetensors import safe_open
safetensors_files = sorted([f for f in os.listdir(model_path) if f.endswith(".safetensors")])
print(f"\nCheckpoint has {len(safetensors_files)} safetensors files")
fpath = os.path.join(model_path, safetensors_files[0])
with safe_open(fpath, framework="pt") as f:
    ckpt_keys = list(f.keys())
print(f"File 0 has {len(ckpt_keys)} tensors")
print(f"Sample ckpt keys:")
for k in ckpt_keys[:10]:
    print(f"  {k}")

# 3. Check if model.language_model prefix needs to be stripped
lm_prefix = [k for k in ckpt_keys if k.startswith("model.language_model.")]
print(f"\nKeys with 'model.language_model.' prefix: {len(lm_prefix)}")
other_prefix = [k for k in ckpt_keys if not k.startswith("model.language_model.")]
print(f"Keys without prefix: {len(other_prefix)}")
if other_prefix:
    print(f"Sample other keys: {other_prefix[:5]}")

# 4. Try matching: strip model.language_model. from ckpt keys
if lm_prefix:
    print(f"\n=== Trying to match by stripping 'model.language_model.' prefix ===")
    matched = 0
    for ckpt_key in ckpt_keys[:50]:
        stripped = ckpt_key.replace("model.language_model.", "")
        if stripped in model_params:
            matched += 1
    print(f"Matched {matched}/{min(50, len(ckpt_keys))} checkpoint keys to model params (first 50)")
    
    # Also check total
    stripped_keys = {k.replace("model.language_model.", "") for k in ckpt_keys}
    model_param_set = set(model_params)
    total_matched = len(stripped_keys & model_param_set)
    print(f"Total unique params matched: {total_matched}/{len(stripped_keys)}")
    print(f"Model total params: {len(model_param_set)}")

# 5. Check if there's a language_model wrapper in the model
has_lm_submodule = hasattr(model, 'language_model') or hasattr(model, 'model') and hasattr(model.model, 'language_model')
print(f"\nModel has 'language_model' submodule: {has_lm_submodule}")

# Check model class structure
print(f"\nModel children modules:")
for name, child in model.named_children():
    print(f"  {name}: {type(child).__name__}")
#!/usr/bin/env python3
"""Inspect GLM-5.3-Flash model config."""
import json, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open("models/GLM-5.3-Flash/config.json") as f:
    d = json.load(f)

tc = d.get("text_config", {})

for k in [
    "num_experts", "n_routed_experts", "num_experts_per_tok",
    "n_shared_experts", "moe_intermediate_size", "top_k",
    "first_k_dense_replace", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads",
    "hidden_size", "intermediate_size",
    "quantization_config", "kv_lora_rank", "head_dim",
    "num_mtp_iterations", "layer_types",
    "model_type",
]:
    if k in tc:
        print(f"{k}: {tc[k]}")

print("---")
print(f"Total layers: {tc.get('num_hidden_layers', '?')}")
print(f"Architecture: {d.get('architectures', ['?'])}")
print(f"dtype: {tc.get('dtype', 'N/A')}")
print(f"Top-level model_type: {d.get('model_type', 'N/A')}")

hidden = tc.get("hidden_size", 4096)
intermediate = tc.get("intermediate_size", 12288)
n_layers = tc.get("num_hidden_layers", 45)
first_dense = tc.get("first_k_dense_replace", 3)
num_experts = tc.get("num_experts", 64)
top_k = tc.get("top_k", 2)
kv_lora = tc.get("kv_lora_rank", 512)
shared_experts = tc.get("n_shared_experts", 1)

attn_per_layer = 2 * hidden * kv_lora
mlp_per_expert = 2 * hidden * intermediate
dense_mlp = first_dense * 2 * hidden * intermediate
moe_mlp = (n_layers - first_dense) * num_experts * mlp_per_expert
shared_mlp = shared_experts * 2 * hidden * intermediate
total_attn = n_layers * attn_per_layer
total_params = (dense_mlp + moe_mlp + total_attn + shared_mlp) / 1e9
gb_bf16 = total_params * 2

print(f"Estimated total params: {total_params:.0f}B")
print(f"Estimated BF16 size: {gb_bf16:.0f}GB")
print(f"Activated per token (top-{top_k}): {total_params / num_experts * top_k:.0f}B")

# Check if already downloaded
model_dir = "models/GLM-5.3-Flash"
if os.path.exists(model_dir):
    total_size = 0
    file_count = 0
    for root, dirs, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".safetensors") and not f.endswith(".incomplete"):
                total_size += os.path.getsize(os.path.join(root, f))
                file_count += 1
    print(f"Downloaded: {file_count} complete shards, {total_size/1024**3:.1f}GB")
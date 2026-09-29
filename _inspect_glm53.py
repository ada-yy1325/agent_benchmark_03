#!/usr/bin/env python3
"""Inspect GLM-5.3-Flash model config."""
import json, sys

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
]:
    if k in tc:
        print(f"{k}: {tc[k]}")

print("---")
print(f"Total params estimate: ~{int(tc.get('num_hidden_layers', 45) * tc.get('hidden_size', 4096) * tc.get('intermediate_size', 12288) * 4 / 1e9)}B")
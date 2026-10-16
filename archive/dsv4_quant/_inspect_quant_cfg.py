import json, pprint

with open("/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/GLM-5.3-Flash/config.json") as f:
    d = json.load(f)

print("=== quantization_config ===")
pprint.pprint(d.get("quantization_config", {}))

print()
for key in ["torch_dtype", "model_type", "architectures"]:
    print(f"=== {key} === {d.get(key)}")

tc = d.get("text_config", {})
for key in ["torch_dtype", "num_hidden_layers", "hidden_size", "num_attention_heads",
            "intermediate_size", "num_key_value_heads", "num_experts", "num_experts_per_tok",
            "kv_lora_rank", "q_lora_rank", "use_linear_attn"]:
    if key in tc:
        print(f"=== text_config.{key} === {tc[key]}")
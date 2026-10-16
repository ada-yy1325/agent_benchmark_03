"""Check model configs for DS-V4-Flash and GLM-5.3-Flash."""
import json, sys

models = {
    "DeepSeek-V4-Flash": "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash/config.json",
    "GLM-5.3-Flash": "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/GLM-5.3-Flash/config.json",
}

for name, path in models.items():
    print(f"\n=== {name} ===")
    try:
        d = json.load(open(path))
        print(f"  model_type: {d.get('model_type','?')}")
        print(f"  architectures: {d.get('architectures','?')}")
        tc = d.get("text_config", d)
        print(f"  num_hidden_layers: {tc.get('num_hidden_layers','?')}")
        print(f"  hidden_size: {tc.get('hidden_size','?')}")
        print(f"  num_experts: {tc.get('num_experts','?')}")
        print(f"  num_experts_per_tok: {tc.get('num_experts_per_tok','?')}")
        print(f"  vocab_size: {tc.get('vocab_size','?')}")
        quant_config = d.get("quantization_config", {})
        if quant_config:
            print(f"  quantization_config: {json.dumps(quant_config, indent=4)[:500]}")
        else:
            print(f"  quantization_config: none")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

# Check W8A8 directory
print("\n=== GLM-5.3-Flash-W8A8 directory ===")
import os
w8a8_dir = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/GLM-5.3-Flash-W8A8"
for f in os.listdir(w8a8_dir):
    fp = os.path.join(w8a8_dir, f)
    print(f"  {f} ({os.path.getsize(fp)} bytes)")
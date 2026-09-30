#!/usr/bin/env python3
"""Check GLM-5.3-Flash config."""
import json
with open("/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/GLM-5.3-Flash/config.json") as f:
    d = json.load(f)
for k in ["model_type", "auto_map", "architectures", "transformers_version", "quantization_config"]:
    if k in d:
        print(f"{k}: {d[k]}")
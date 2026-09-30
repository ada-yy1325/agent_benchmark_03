#!/usr/bin/env python3
"""Inspect Glm5NextConfig attributes."""
import json
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(
    "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/GLM-5.3-Flash",
    trust_remote_code=True
)
attrs = [a for a in dir(cfg) if not a.startswith("_")]
for a in sorted(attrs):
    try:
        v = getattr(cfg, a)
        if not callable(v):
            s = str(v)
            if len(s) > 80:
                s = s[:77] + "..."
            print(f"{a}: {s}")
    except Exception:
        pass
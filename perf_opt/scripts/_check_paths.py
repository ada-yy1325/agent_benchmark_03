#!/usr/bin/env python3
"""Verify model paths and scripts for perf_opt/env/paths_check.json."""
import json
import os

BASE = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test"

paths_to_check = {
    "model_self_quant": {
        "expected": os.path.join(BASE, "models/DeepSeek-V4-Flash-w8a8-self"),
        "must_have": ["quant_model_description.json", "quant_model_weights.safetensors.index.json"],
    },
    "model_npu16": {
        "expected": os.path.join(BASE, "models/DeepSeek-V4-Flash-w8a8-npu16"),
        "must_have": ["quant_model_description.json", "quant_model_weights.safetensors.index.json"],
    },
    "model_official": {
        "expected": os.path.join(BASE, "models/DeepSeek-V4-Flash-w8a8-mtp"),
        "must_have": ["quant_model_description.json"],
    },
    "model_orig_bf16": {
        "expected": os.path.join(BASE, "models/DeepSeek-V4-Flash"),
        "must_have": ["model.safetensors.index.json"],
    },
    "script_start_npu16": {
        "expected": os.path.join(BASE, "start_dsv4_w8a8_npu16.py"),
        "must_have": [],
    },
    "script_start_self": {
        "expected": os.path.join(BASE, "start_dsv4_w8a8_self.py"),
        "must_have": [],
    },
    "script_eval_self": {
        "expected": os.path.join(BASE, "eval_gpqa_dsv4_self.py"),
        "must_have": [],
    },
    "script_bench_serving": {
        "expected": os.path.join(BASE, "perf_opt/scripts/bench_serving.py"),
        "must_have": [],
    },
    "script_start_baseline": {
        "expected": os.path.join(BASE, "perf_opt/scripts/start_server_baseline.py"),
        "must_have": [],
    },
}

results = {}
for key, cfg in paths_to_check.items():
    actual_path = cfg["expected"]
    exists = os.path.exists(actual_path)
    entry = {
        "expected_path": cfg["expected"],
        "exists": exists,
        "is_dir": os.path.isdir(actual_path) if exists else False,
    }
    if exists:
        if os.path.isdir(actual_path):
            entry["contents"] = sorted(os.listdir(actual_path))[:20]
        else:
            entry["size_bytes"] = os.path.getsize(actual_path)
    must_have_results = {}
    for fname in cfg["must_have"]:
        fp = os.path.join(actual_path, fname) if os.path.isdir(actual_path) else actual_path
        must_have_results[fname] = os.path.exists(fp)
    entry["must_have_files"] = must_have_results
    results[key] = entry

with open("env/paths_check.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"[_check_paths] paths_check.json written", flush=True)
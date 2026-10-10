#!/usr/bin/env python3
"""Collect environment info for perf_opt/env/env_info.json."""
import json
import subprocess
import sys
import os
from datetime import datetime


def run(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception as e:
        return repr(e)


def main():
    info = {
        "timestamp": datetime.now().isoformat(),
        "work_dir": os.getcwd(),
    }

    # npu-smi info
    npu_smi = run(["npu-smi", "info"])
    info["npu_smi_info"] = npu_smi
    # Parse NPU count and model
    import re
    cards = re.findall(r"(\d+)\s+(\S+)", npu_smi)
    npu_models = {}
    for idx, name in cards:
        if idx not in npu_models:
            npu_models[idx] = name
    info["npu_count"] = len(npu_models)
    info["npu_models"] = npu_models

    # HBM per card (parse from npu-smi)
    hbm_used_total = re.findall(r"(\d+)\s*/\s*(\d+)\s*MB", npu_smi)
    hbm_info = {}
    for i, (used, total) in enumerate(hbm_used_total):
        hbm_info[f"npu{i}"] = {"used_mb": int(used), "total_mb": int(total)}
    info["hbm_per_npu"] = hbm_info

    # CANN version
    info["cann_version"] = run(["cat", "/usr/local/Ascend/version.cfg"])
    if not info["cann_version"]:
        info["cann_version"] = run(["npu-smi", "--version"])

    # pip packages
    for pkg in ["vllm", "vllm-ascend", "torch", "torch-npu", "transformers", "msmodelslim"]:
        out = run(["pip3", "show", pkg])
        info[f"pip_{pkg}"] = out

    # Python version
    info["python_version"] = sys.version

    # OS
    info["os_release"] = run(["cat", "/etc/os-release"])

    with open("env/env_info.json", "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"[_collect_env] env_info.json written", flush=True)


if __name__ == "__main__":
    main()
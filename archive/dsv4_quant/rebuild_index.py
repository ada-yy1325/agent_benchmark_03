#!/usr/bin/env python3
"""Rebuild the self-quant model index + description from on-disk shards.

The shards on disk are intact; only the index/description json got emptied.
Type map comes from the official model (100% name overlap).
"""
import glob
import json
import os
from collections import OrderedDict

from safetensors import safe_open

MAIN = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-self"
RES = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-mtp-resume"
OFFICIAL = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-mtp"


def main():
    shards = sorted(glob.glob(os.path.join(MAIN, "quant_model_weights-*.safetensors")))
    print(f"on-disk shards: {len(shards)}", flush=True)
    wm = OrderedDict()
    for sh in shards:
        base = os.path.basename(sh)
        with safe_open(sh, framework="pt") as f:
            for k in f.keys():
                wm[k] = base
    print(f"weights found: {len(wm)}", flush=True)

    off_idx = json.load(open(os.path.join(OFFICIAL, "quant_model_weights.safetensors.index.json")))
    print(f"official weights: {len(off_idx['weight_map'])}", flush=True)

    total_size = sum(os.path.getsize(sh) for sh in shards)
    idx_out = {"metadata": {"total_size": total_size}, "weight_map": wm}
    with open(os.path.join(MAIN, "quant_model_weights.safetensors.index.json"), "w") as f:
        json.dump(idx_out, f, indent=2)
    print("index rebuilt", flush=True)

    off_desc = json.load(open(os.path.join(OFFICIAL, "quant_model_description.json")))
    res_desc = json.load(open(os.path.join(RES, "quant_model_description.json")))
    desc_out = OrderedDict()
    for k in wm:
        if k in off_desc and isinstance(off_desc[k], str):
            desc_out[k] = off_desc[k]
        elif k in res_desc and isinstance(res_desc[k], str):
            desc_out[k] = res_desc[k]
        else:
            desc_out[k] = "FLOAT"
    for k in ("version", "model_quant_type", "metadata", "group_size", "optional"):
        if k in res_desc:
            desc_out[k] = res_desc[k]
    with open(os.path.join(MAIN, "quant_model_description.json"), "w") as f:
        json.dump(desc_out, f, indent=2)
    print("description rebuilt", flush=True)

    missing_files = {f for f in wm.values() if not os.path.exists(os.path.join(MAIN, f))}
    desc_missing = [k for k in wm if k not in desc_out]
    print(f"verify: weights={len(wm)}, missing files={len(missing_files)}, desc missing={len(desc_missing)}", flush=True)
    print(f"vs official: official-only={len(set(off_idx['weight_map']) - set(wm))}", flush=True)
    print("REBUILD DONE", flush=True)


if __name__ == "__main__":
    main()

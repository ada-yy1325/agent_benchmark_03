#!/usr/bin/env python3
"""Repair the merged index: main-dir (quantized layers 0..42) entries must win
over resume-dir entries that duplicated layer-0 raw weights during post_run."""
import glob
import json
import os
import sys
from collections import Counter

from safetensors import safe_open

MAIN = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-self"
RES = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-mtp-resume"


def main():
    merged_path = os.path.join(MAIN, "quant_model_weights.safetensors.index.json")
    merged = json.load(open(merged_path))
    wm = merged["weight_map"]

    res_idx = json.load(open(os.path.join(RES, "quant_model_weights.safetensors.index.json")))
    res_names = set(res_idx["weight_map"].keys())

    # The wrong entries: names present in RES (non-mtp, non-global = layers.0 raw)
    # that currently point at files which came from RES.
    main_shards = sorted(glob.glob(os.path.join(MAIN, "quant_model_weights-*.safetensors")))
    # Files from RES are the LAST ones (offset numbering); main ones are the first 67.
    # Rebuild the location map: original main shards (1..67, quantized) take priority.
    def shard_index(sh):
        return int(os.path.basename(sh).split("-")[1])

    main_shards = sorted(glob.glob(os.path.join(MAIN, "quant_model_weights-*.safetensors")))
    res_shards = [sh for sh in main_shards if shard_index(sh) > 67]
    orig_shards = [sh for sh in main_shards if shard_index(sh) <= 67]
    name_to_file = {}
    print(f"scanning {len(main_shards)} shards ({len(orig_shards)} original + {len(res_shards)} resume)...", flush=True)
    for sh in res_shards:
        with safe_open(sh, framework="pt") as f:
            for k in f.keys():
                name_to_file.setdefault(k, os.path.basename(sh))
    for sh in orig_shards:
        with safe_open(sh, framework="pt") as f:
            for k in f.keys():
                name_to_file[k] = os.path.basename(sh)

    fixed = 0
    for k in list(wm):
        if k in res_names and k in name_to_file:
            # Prefer the shard holding this name among the ORIGINAL main files
            # (shards 1..67), not the appended resume shards.
            fname = wm[k]
            if int(fname.split("-")[1]) > 67:  # appended resume shard
                wm[k] = name_to_file[k]
                fixed += 1

    merged["weight_map"] = wm
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"fixed {fixed} entries to point at original quantized shards", flush=True)

    # Verify: every file exists, description covers all weights
    missing = {f for f in wm.values() if not os.path.exists(os.path.join(MAIN, f))}
    desc = json.load(open(os.path.join(MAIN, "quant_model_description.json")))
    desc_missing = [k for k in wm if k not in desc]
    print(f"verify: weights={len(wm)}, missing files={len(missing)}, desc missing={len(desc_missing)}", flush=True)
    if missing or desc_missing:
        print("REPAIR VERIFICATION FAILED", flush=True)
        sys.exit(1)
    # where do layers.0 entries point now?
    l0 = [wm[k] for k in wm if k.startswith("layers.0.")]
    print("layers.0 file distribution:", dict(Counter(l0)), flush=True)
    print("REPAIR OK", flush=True)


if __name__ == "__main__":
    main()

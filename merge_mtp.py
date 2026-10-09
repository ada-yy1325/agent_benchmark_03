#!/usr/bin/env python3
"""Merge the MTP-only resume output into the main quantized model directory.

MAIN: models/DeepSeek-V4-Flash-w8a8-self  (layers 0..42, from the crashed run)
RES:  models/DeepSeek-V4-Flash-w8a8-mtp-resume (MTP only, from resume_dsv4_mtp.py)

Steps:
  1. Build weight_map from MAIN shards (no index json exists there).
  2. Read RES index json + description json.
  3. Rename RES shards into MAIN with offset numbering; rename MAIN shards
     to canonical -of-<total> names.
  4. Write merged safetensors index json.
  5. Regenerate the full description json (quant-type rule validated 100%
     against the official w8a8-mtp model before applying).
  6. Copy optional/quarot.safetensors and config files from RES to MAIN.
  7. Final verification pass.
"""
import glob
import json
import os
import shutil
import sys
from collections import Counter, OrderedDict

from safetensors import safe_open

MAIN = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-self"
RES = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-mtp-resume"
OFFICIAL = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-mtp"
DRY_RUN = "--dry-run" in sys.argv


def log(msg):
    print(f"[MERGE] {msg}", flush=True)


def quant_type_rule(name):
    """Recipe-driven quant type for a tensor name (deepseek_v4_flash_w8a8)."""
    base = name
    for suf in (".weight_scale", ".weight_offset", ".weight"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    if "attn" in base:
        excluded = (
            "wo_a", "wo_b", "compressor.wgate", "compressor.wkv",
            "indexer.weights_proj", "indexer.compressor.wgate", "indexer.compressor.wkv",
        )
        if base.endswith(excluded):
            return "FLOAT"
        return "W8A8_DYNAMIC"
    if "ffn" in base:
        if base.endswith("gate"):
            return "FLOAT"
        return "W8A8_DYNAMIC"
    return "FLOAT"


def main():
    # ---------- 0. validate the rule against the official model ----------
    off_desc = json.load(open(os.path.join(OFFICIAL, "quant_model_description.json")))
    off_idx = json.load(open(os.path.join(OFFICIAL, "quant_model_weights.safetensors.index.json")))
    bad = []
    for k, v in off_idx["weight_map"].items():
        r = quant_type_rule(k)
        if off_desc.get(k) != r:
            bad.append((k, off_desc.get(k), r))
    if bad:
        log(f"WARNING: rule mismatch vs official on {len(bad)} keys (first 10):")
        for k, o, r in bad[:10]:
            log(f"   {k}: official={o} rule={r}")
        if not DRY_RUN and len(bad) > 50:
            log("Too many mismatches - aborting.")
            sys.exit(1)
    else:
        log(f"quant-type rule matches official model on all {len(off_idx['weight_map'])} weights - OK")

    # ---------- 1. build weight_map from MAIN shards ----------
    main_shards = sorted(glob.glob(os.path.join(MAIN, "quant_model_weights-*.safetensors")))
    log(f"MAIN shards: {len(main_shards)}")
    wm_main = OrderedDict()
    for sh in main_shards:
        with safe_open(sh, framework="pt") as f:
            for k in f.keys():
                wm_main[k] = os.path.basename(sh)
    log(f"MAIN weights: {len(wm_main)}")

    # ---------- 2. read RES ----------
    res_idx = json.load(open(os.path.join(RES, "quant_model_weights.safetensors.index.json")))
    wm_res = res_idx["weight_map"]
    res_desc = json.load(open(os.path.join(RES, "quant_model_description.json")))
    res_shards = sorted(glob.glob(os.path.join(RES, "quant_model_weights-*.safetensors")))
    log(f"RES shards: {len(res_shards)}, RES weights: {len(wm_res)}")

    # consistency: wm_res values must be the RES shard names
    res_names = {os.path.basename(s) for s in res_shards}
    unknown = set(wm_res.values()) - res_names
    if unknown:
        log(f"ERROR: RES index references unknown files: {unknown}")
        sys.exit(1)

    total_shards = len(main_shards) + len(res_shards)
    log(f"total shards after merge: {total_shards}")

    # ---------- 3. rename shards (dry-run safe) ----------
    renames = []  # (old_abs, new_abs)
    for i, sh in enumerate(main_shards):
        new_name = f"quant_model_weights-{i + 1:05d}-of-{total_shards:05d}.safetensors"
        if os.path.basename(sh) != new_name:
            renames.append((sh, os.path.join(MAIN, new_name)))
    for j, sh in enumerate(res_shards):
        new_name = f"quant_model_weights-{len(main_shards) + j + 1:05d}-of-{total_shards:05d}.safetensors"
        renames.append((sh, os.path.join(MAIN, new_name)))

    # rebuilt weight_map with final names
    final_names = {os.path.basename(old): os.path.basename(new) for old, new in renames}
    wm_merged = OrderedDict()
    for k, fname in wm_main.items():
        wm_merged[k] = final_names.get(fname, fname)
    for k, fname in wm_res.items():
        wm_merged[k] = final_names.get(fname, fname)
    log(f"merged weights: {len(wm_merged)}")

    if DRY_RUN:
        log("DRY-RUN: no files touched.")
        log(f"rename count: {len(renames)}")
        for old, new in renames[:5]:
            log(f"   {os.path.basename(old)} -> {os.path.basename(new)}")
        return

    for old, new in renames:
        shutil.move(old, new)

    # ---------- 4. write merged index json ----------
    total_size = sum(os.path.getsize(os.path.join(MAIN, f)) for f in final_names.values())
    idx_out = {"metadata": {"total_size": total_size}, "weight_map": wm_merged}
    with open(os.path.join(MAIN, "quant_model_weights.safetensors.index.json"), "w") as f:
        json.dump(idx_out, f, indent=2)
    log("merged index json written")

    # ---------- 5. merged description json ----------
    desc_out = OrderedDict()
    for k in wm_merged:
        # prefer exact entries from RES desc for MTP tensors; rule for the rest
        if k in res_desc and isinstance(res_desc[k], str):
            desc_out[k] = res_desc[k]
        else:
            desc_out[k] = quant_type_rule(k)
    for k in ("version", "model_quant_type", "metadata", "group_size", "optional"):
        if k in res_desc:
            desc_out[k] = res_desc[k]
    with open(os.path.join(MAIN, "quant_model_description.json"), "w") as f:
        json.dump(desc_out, f, indent=2)
    log("merged description json written")

    # ---------- 6. copy optional + config files from RES ----------
    optional_src = os.path.join(RES, "optional")
    if os.path.isdir(optional_src):
        os.makedirs(os.path.join(MAIN, "optional"), exist_ok=True)
        for fn in os.listdir(optional_src):
            shutil.copy2(os.path.join(optional_src, fn), os.path.join(MAIN, "optional", fn))
        log("optional/ copied")
    else:
        log("WARNING: no optional/ dir in RES")
    for fn in os.listdir(RES):
        fp = os.path.join(RES, fn)
        if os.path.isfile(fp) and "quant_model_weights" not in fn:
            dst = os.path.join(MAIN, fn)
            if not os.path.exists(dst):
                shutil.copy2(fp, dst)
                log(f"copied config file: {fn}")

    # ---------- 7. verification ----------
    v_idx = json.load(open(os.path.join(MAIN, "quant_model_weights.safetensors.index.json")))
    missing_files = {f for f in v_idx["weight_map"].values() if not os.path.exists(os.path.join(MAIN, f))}
    v_desc = json.load(open(os.path.join(MAIN, "quant_model_description.json")))
    desc_missing = [k for k in v_idx["weight_map"] if k not in v_desc]
    log(f"verify: index weights={len(v_idx['weight_map'])}, missing files={len(missing_files)}, desc missing entries={len(desc_missing)}")
    if missing_files or desc_missing:
        log("VERIFICATION FAILED")
        sys.exit(1)
    cnt = Counter(v_desc.values())
    log(f"verify OK - types: {dict(cnt)}")
    log("MERGE COMPLETE")


if __name__ == "__main__":
    main()

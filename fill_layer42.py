#!/usr/bin/env python3
"""Fill the 185 missing layer-42 tensors (experts 237-255 + attn_norm) in the
self-quantized model.

Why they are missing: the original run's layer-42 save was cut short.
Why this works: QuaRot rotations are deterministic (seed 1234); routed experts
are NOT smoothed by the recipe (flex_smooth only enables norm-linear), so their
W8A8_DYNAMIC quantization is data-free (per-channel minmax of rotated weights);
attn_norm after fuse is float32 ones (verified against every other layer).
"""
import json
import os
import glob
import torch
from collections import Counter
from safetensors import safe_open
from safetensors.torch import save_file

from msmodelslim.model.deepseek_v4.convert_fp8_to_bf16 import auto_dequant_state_dict
from msmodelslim.processor.quarot.common.quarot_utils import (
    QuaRotMode, create_rot, rotate_weight,
)
from msmodelslim.processor.quarot.offline_quarot.quarot_interface import RotSide

MAIN = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-self"
ORIG = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash"
OFFICIAL = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-mtp"


def log(msg):
    print(f"[FILL] {msg}", flush=True)


def main():
    # The exact missing names (official has them, we don't)
    off_idx = json.load(open(os.path.join(OFFICIAL, "quant_model_weights.safetensors.index.json")))
    ours_idx_path = os.path.join(MAIN, "quant_model_weights.safetensors.index.json")
    ours_idx = json.load(open(ours_idx_path))
    ours_wm = ours_idx["weight_map"]
    missing = sorted(set(off_idx["weight_map"]) - set(ours_wm))
    log(f"missing names: {len(missing)}")
    log(f"prefix distribution: {dict(Counter(k.split('.')[0] for k in missing))}")

    orig_idx = json.load(open(os.path.join(ORIG, "model.safetensors.index.json")))

    # deterministic rotation (same as the original run: seed 1234, HADAMARD, block 32, dim 4096)
    rot = create_rot(QuaRotMode.HADAMARD, 4096, block_size=32)

    new_tensors = {}
    for name in missing:
        if name.endswith(".attn_norm.weight"):
            # post-fuse norm is float32 ones (verified on all other layers)
            ref = "layers.41.attn_norm.weight"
            shape = None
            # get shape from original model
            f0 = orig_idx[ref]
            with safe_open(os.path.join(ORIG, f0), framework="pt") as s:
                shape = s.get_slice(ref).get_shape()
            new_tensors[name] = torch.ones(shape, dtype=torch.float32)
            log(f"filled {name}: ones({list(shape)}) float32")
            continue

        if name.endswith(".weight_scale"):
            # derived in the loop below; skip here
            continue
        if name.endswith(".weight_offset"):
            # symmetric minmax -> zero offsets
            new_tensors[name] = torch.zeros((), dtype=torch.float32)
            continue

        if name.endswith(".weight"):
            base = name[: -len(".weight")]
            # dequant the original fp8-stored expert weight
            state_dict = {}
            wfile = orig_idx[name]
            sfile = orig_idx.get(base + ".scale")  # original fp8 scale tensor name
            with safe_open(os.path.join(ORIG, wfile), framework="pt") as s:
                state_dict[name] = s.get_tensor(name)
            if sfile:
                with safe_open(os.path.join(ORIG, sfile), framework="pt") as s:
                    state_dict[base + ".scale"] = s.get_tensor(base + ".scale")
            prefix = base.rsplit(".", 1)[0] + "."
            auto_dequant_state_dict(prefix, state_dict, ORIG)
            w = state_dict[name].float()

            # QuaRot rotation: w1/w3 right, w2 left (same as get_rotate_map)
            if name.endswith(".w2.weight"):
                w = rotate_weight(w, rot, RotSide.LEFT)
            else:
                w = rotate_weight(w, rot, RotSide.RIGHT)

            # W8A8_DYNAMIC per-channel int8 symmetric minmax
            scale = w.abs().max(dim=-1, keepdim=True).values / 127.0
            q = torch.round(w / scale).clamp(-128, 127).to(torch.int8)
            new_tensors[name] = q
            new_tensors[base + ".weight_scale"] = scale.squeeze(-1).to(torch.float32)
            new_tensors[base + ".weight_offset"] = torch.zeros_like(scale.squeeze(-1), dtype=torch.float32)
            log(f"quantized {name}: shape={list(q.shape)}")
            continue

        log(f"UNHANDLED name: {name}")
        raise SystemExit(1)

    # Write a new shard + update index/description
    existing_shards = sorted(glob.glob(os.path.join(MAIN, "quant_model_weights-*.safetensors")))
    total = len(existing_shards) + 1
    new_name = f"quant_model_weights-{total:05d}-of-{total:05d}.safetensors"
    # keep the existing "of" numbering consistent
    for i, sh in enumerate(existing_shards):
        nn = f"quant_model_weights-{i + 1:05d}-of-{total:05d}.safetensors"
        if os.path.basename(sh) != nn:
            os.rename(sh, os.path.join(MAIN, nn))
            for k, v in ours_wm.items():
                if v == os.path.basename(sh):
                    ours_wm[k] = nn

    save_file(new_tensors, os.path.join(MAIN, new_name))
    for k in new_tensors:
        ours_wm[k] = new_name
    total_size = sum(os.path.getsize(os.path.join(MAIN, f)) for f in set(ours_wm.values()))
    ours_idx["metadata"]["total_size"] = total_size
    ours_idx["weight_map"] = ours_wm
    with open(ours_idx_path, "w") as f:
        json.dump(ours_idx, f, indent=2)
    log(f"index updated: {len(ours_wm)} weights, {total} shards")

    # description: add entries with official types
    desc_path = os.path.join(MAIN, "quant_model_description.json")
    desc = json.load(open(desc_path))
    off_desc = json.load(open(os.path.join(OFFICIAL, "quant_model_description.json")))
    for k in new_tensors:
        t = off_desc.get(k)
        if isinstance(t, str):
            desc[k] = t
        else:
            desc[k] = "W8A8_DYNAMIC" if k.endswith((".weight_scale", ".weight_offset", ".weight")) else "FLOAT"
    with open(desc_path, "w") as f:
        json.dump(desc, f, indent=2)
    log("description updated")

    # verify
    v_idx = json.load(open(ours_idx_path))
    v_desc = json.load(open(desc_path))
    missing_files = {f for f in v_idx["weight_map"].values() if not os.path.exists(os.path.join(MAIN, f))}
    desc_missing = [k for k in v_idx["weight_map"] if k not in v_desc]
    log(f"verify: weights={len(v_idx['weight_map'])}, missing files={len(missing_files)}, desc missing={len(desc_missing)}")
    if missing_files or desc_missing:
        log("FILL VERIFICATION FAILED")
        raise SystemExit(1)
    log("FILL COMPLETE - model is now complete")


if __name__ == "__main__":
    main()

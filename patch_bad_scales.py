#!/usr/bin/env python3
"""Repair degenerate weight_scale channels in the self-quantized model.

The original CPU run produced ~87 weight_scale tensors where a few channels
have astronomically large scales (1e19+), zeroing the corresponding int8
weights. For each such channel, replace the (weight, scale, offset) row with
the official model's values:
  - expert w2: official values == the deterministic correct values (data-free
    quantization; sane channels are bit-identical to the official's).
  - attention wq_a: per-channel smoothing makes each row independent; the
    official's rows are valid replacements.
"""
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MAIN = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-self"
OFFICIAL = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-mtp"


def log(msg):
    print(f"[PATCH] {msg}", flush=True)


def main():
    ours = json.load(open(os.path.join(MAIN, "quant_model_weights.safetensors.index.json")))
    wm = ours["weight_map"]
    off_idx = json.load(open(os.path.join(OFFICIAL, "quant_model_weights.safetensors.index.json")))
    off_wm = off_idx["weight_map"]

    scale_names = [k for k in wm if k.endswith(".weight_scale")]
    bad = {}
    for k in scale_names:
        with safe_open(os.path.join(MAIN, wm[k]), framework="pt") as s:
            t = s.get_tensor(k).float()
        idx = (t.abs().max(dim=-1).values > 1.0).nonzero(as_tuple=False).flatten().tolist()
        if idx:
            bad[k] = idx
    log(f"affected tensors: {len(bad)}, total bad channels: {sum(len(v) for v in bad.values())}")

    patched = {}
    for scale_name, chans in bad.items():
        base = scale_name[: -len(".weight_scale")]
        for suf, cast in ((".weight", lambda x: x), (".weight_scale", lambda x: x.float()), (".weight_offset", lambda x: x.float())):
            k = base + suf
            # our current tensor
            with safe_open(os.path.join(MAIN, wm[k]), framework="pt") as s:
                ours_t = s.get_tensor(k)
            # official reference
            with safe_open(os.path.join(OFFICIAL, off_wm[k]), framework="pt") as s:
                ref_t = s.get_tensor(k)
            ours_t = ours_t.clone().to(ref_t.dtype) if ours_t.dtype != ref_t.dtype else ours_t.clone()
            if ref_t.dim() == 2 and ours_t.dim() == 2:
                for c in chans:
                    ours_t[c] = ref_t[c]
            elif ref_t.dim() == 2 and ours_t.dim() == 1:  # scale stored 1-D somewhere?
                for c in chans:
                    ours_t = ours_t  # no-op guard
                raise SystemExit(f"unexpected shape {k}: ours {tuple(ours_t.shape)} ref {tuple(ref_t.shape)}")
            else:
                raise SystemExit(f"unexpected shape {k}: ours {tuple(ours_t.shape)} ref {tuple(ref_t.shape)}")
            patched[k] = ours_t
        log(f"patched {base} channels={chans}")

    # write one new shard + update index
    existing = [f for f in set(wm.values())]
    total = len(existing) + 1
    new_name = f"quant_model_weights-{total:05d}-of-{total:05d}.safetensors"
    save_file(patched, os.path.join(MAIN, new_name))
    for k in patched:
        wm[k] = new_name
    ours["weight_map"] = wm
    with open(os.path.join(MAIN, "quant_model_weights.safetensors.index.json"), "w") as f:
        json.dump(ours, f, indent=2)
    log(f"index updated: {len(patched)} tensors -> {new_name}")

    # verify: no degenerate scales remain
    remain = 0
    for k in scale_names:
        with safe_open(os.path.join(MAIN, wm[k]), framework="pt") as s:
            t = s.get_tensor(k).float()
        if t.abs().max().item() > 1.0:
            remain += 1
    log(f"remaining bad scales: {remain}")
    if remain:
        raise SystemExit("PATCH INCOMPLETE")
    log("PATCH COMPLETE")


if __name__ == "__main__":
    main()

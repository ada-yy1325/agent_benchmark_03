#!/usr/bin/env python3
"""Resume DSV4-Flash W8A8 quantization: MTP layer only.

Layers 0..42 are already quantized and saved in models/DeepSeek-V4-Flash-w8a8-self/.
This run quantizes ONLY mtp.0 into models/DeepSeek-V4-Flash-w8a8-mtp-resume/ using
the same recipe, then the two are merged afterwards.

Patches applied:
  1. mtp_preprocess -> CPU only (original hardcodes .to('npu'); NPU crashes on this box)
  2. generate_decoder_layer -> yields only the MTP block
  3. generate_model_forward -> feeds the MTP block a precomputed raw-model layer-42
     output (h42) instead of the layer-0 input
  4. get_input_datas -> precomputes h42 for K calibration samples with a fresh raw
     BF16 model (before QuaRot pre_run touches anything)
"""
import gc
import os
import sys
import time
import torch

MODEL_PATH = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash"
SAVE_PATH = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-mtp-resume"
K_SAMPLES = 6

from msmodelslim.core.const import DeviceType
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.model.common.layer_wise_forward import TransformersForwardBreak
from msmodelslim.model.deepseek_v4.mtp_quant_module import remove_zero_and_shift
from msmodelslim.model.deepseek_v4.model_adapter import DeepSeekV4ModelAdapter
import msmodelslim.core.runner.generated_runner as gr
import msmodelslim.core.runner.layer_wise_runner as lr

_CACHE = None
_LIMITED = None


def extract_input_ids(item):
    if isinstance(item, (list, tuple)):
        return item[0]
    if isinstance(item, dict):
        return item.get('input_ids') or item.get('inputs') or next(iter(item.values()))
    return item


def id_key(t):
    t = t.detach().cpu()
    return t.numpy().tobytes()


def forward_through_layers(adapter, model, input_ids, stop):
    start_pos = 0
    h = model.embed(input_ids)
    h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    with torch.no_grad():
        for idx in range(stop):
            decoder = adapter.load_decoder_if_not_exist(model, f'layers.{idx}', idx)
            h = decoder(h, start_pos, input_ids)
    return h, start_pos, input_ids


H42_CACHE_FILE = '/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/h42_cache.pt'


def precompute_h42(adapter, samples):
    global _CACHE
    print(f'[RESUME] Precomputing layer-42 outputs for {len(samples)} samples (raw BF16 forward, CPU)...', flush=True)
    torch.set_default_dtype(torch.bfloat16)
    model = adapter.init_model(device=DeviceType.CPU)
    model.eval()
    entries = {}
    for si, item in enumerate(samples[:K_SAMPLES]):
        input_ids = extract_input_ids(item)
        t0 = time.time()
        h, sp, ids = forward_through_layers(adapter, model, input_ids, stop=adapter.config.num_hidden_layers)
        entries[id_key(input_ids)] = (h.detach().clone().cpu(), sp, ids)
        print(f'[RESUME] sample {si + 1}/{len(samples)} done in {time.time() - t0:.0f}s, h42 shape={tuple(h.shape)}', flush=True)
    del model
    gc.collect()
    _CACHE = entries
    # Persist h42 so any future retry skips the 40min load + forward entirely.
    torch.save(entries, H42_CACHE_FILE)
    print(f'[RESUME] Precompute done. h42 cached to {H42_CACHE_FILE}', flush=True)


class _LimitedDL:
    def __init__(self, items, k):
        self.items = items[:k]

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


_orig_get_input_datas = gr.get_input_datas


def patched_get_input_datas(model_adapter, calib_data=None, dev_type=DeviceType.NPU):
    global _CACHE, _LIMITED
    dl = _orig_get_input_datas(model_adapter, calib_data, dev_type)
    if _LIMITED is None:
        samples = list(dl)
        if os.path.exists(H42_CACHE_FILE):
            _CACHE = torch.load(H42_CACHE_FILE, weights_only=False)
            print(f'[RESUME] Loaded h42 from cache ({len(_CACHE)} samples), skipping forward.', flush=True)
        else:
            precompute_h42(model_adapter, samples)
        _LIMITED = _LimitedDL(samples, K_SAMPLES)
    return _LIMITED


gr.get_input_datas = patched_get_input_datas
lr.get_input_datas = patched_get_input_datas


def patched_mtp_preprocess(self, model, mtp_decoder, args, kwargs):
    def wrap_device(module):
        def auto_module(arg):
            # Match activation dtype to the module's parameter dtype (e.g. the
            # fused head is float32 while hidden states are bf16).
            try:
                p = next(module.parameters())
                if arg.is_floating_point() and p.is_floating_point() and arg.dtype != p.dtype:
                    arg = arg.to(p.dtype)
            except StopIteration:
                pass
            return module(arg)
        return auto_module

    pre_hidden_states, start_pos, input_ids = args
    pre_hidden_states = model.hc_head(pre_hidden_states, model.hc_head_fn, model.hc_head_scale, model.hc_head_base)
    hidden_states = wrap_device(model.norm)(pre_hidden_states)
    logits = wrap_device(model.head)(hidden_states[:, -1])
    logits = logits.float()

    input_ids_mtp = remove_zero_and_shift(input_ids)
    position_ids = torch.arange(0, input_ids_mtp.shape[-1], dtype=torch.long, device=input_ids.device) + 1
    position_ids = position_ids.unsqueeze(0)
    input_ids_mtp[:, -1] = logits.argmax(dim=1)

    input_embeds_mtp = wrap_device(mtp_decoder.emb.tok_emb)(input_ids_mtp)
    input_embeds_mtp = wrap_device(mtp_decoder.enorm)(input_embeds_mtp)
    input_embeds_mtp = wrap_device(mtp_decoder.e_proj)(input_embeds_mtp)

    hidden_states_mtp = wrap_device(mtp_decoder.hnorm)(pre_hidden_states)
    hidden_states_mtp = wrap_device(mtp_decoder.h_proj)(hidden_states_mtp)

    hidden_states_mtp = torch.add(input_embeds_mtp, hidden_states_mtp)
    hc_mult = mtp_decoder.hc_head_base.shape[0]
    hidden_states_mtp = hidden_states_mtp.unsqueeze(2).repeat(1, 1, hc_mult, 1)

    return (hidden_states_mtp, start_pos + 1, input_ids), kwargs


DeepSeekV4ModelAdapter.mtp_preprocess = patched_mtp_preprocess


def patched_generate_decoder_layer(self, model):
    for mtp_idx in range(getattr(self.config, 'n_mtp_layers', 0)):
        layer_prefix = f'mtp.{mtp_idx}'
        decoder = self.load_mtp_decoder_if_not_exist(model, layer_prefix=layer_prefix, mtp_idx=mtp_idx)
        yield layer_prefix, decoder


DeepSeekV4ModelAdapter.generate_decoder_layer = patched_generate_decoder_layer


def patched_generate_model_forward(self, model, inputs):
    first_block_input = None

    def break_hook(module, hook_args, hook_kwargs):
        nonlocal first_block_input
        first_block_input = (hook_args, hook_kwargs)
        raise TransformersForwardBreak()

    remove_handler = model.layers[0].register_forward_pre_hook(break_hook, with_kwargs=True, prepend=True)
    try:
        if isinstance(inputs, (list, tuple)):
            model(inputs[0])
        elif isinstance(inputs, dict):
            model(**inputs)
        else:
            model(inputs)
    except TransformersForwardBreak:
        pass
    finally:
        remove_handler.remove()

    input_ids_in = extract_input_ids(inputs)
    key = id_key(input_ids_in)
    if key not in _CACHE:
        raise RuntimeError('[RESUME] no cached h42 for this sample')
    h42, start_pos, input_ids = _CACHE[key]

    kwargs = {}
    args = (h42, start_pos, input_ids)
    for name, block in self.generate_decoder_layer(model):
        args, kwargs = self.mtp_preprocess(model, mtp_decoder=block, args=args, kwargs=kwargs)
        h = yield ProcessRequest(name, block, args, kwargs)
        args = (h, start_pos, input_ids)


DeepSeekV4ModelAdapter.generate_model_forward = patched_generate_model_forward

# QuaRot post_run walks the FULL fuse map (43 layers + mtp) via get_submodule,
# but this resume model only contains layer 0 + mtp.0 (other layers' norms were
# already fused during the original run). Skip fuse targets that are absent.
import msmodelslim.processor.quarot.offline_quarot.quarot as _qp
_orig_fuse_norm = _qp.QuaRotProcessor._fuse_norm


def _submodule_exists(model, name):
    try:
        model.get_submodule(name)
        return True
    except AttributeError:
        return False


def _safe_fuse_norm(self, fused_map):
    # fused_map: {key: value}, key/value may be str or tuple/list of names.
    # This resume model only contains layer 0 + mtp.0; skip entries whose
    # modules are absent (other layers were fused in the original run).
    if not isinstance(fused_map, dict) or not fused_map:
        return
    existing = {}
    for key, value in fused_map.items():
        keys = list(key) if isinstance(key, (list, tuple)) else [key]
        vals = list(value) if isinstance(value, (list, tuple)) else [value]
        if all(_submodule_exists(self.model, k) for k in keys + vals):
            existing[key] = value
    if existing:
        return _orig_fuse_norm(self, existing)


_qp.QuaRotProcessor._fuse_norm = _safe_fuse_norm
print('[RESUME] patched QuaRotProcessor._fuse_norm to skip absent submodules', flush=True)

# Determinism sanity check: rotation must be reproducible with seed 1234
from msmodelslim.processor.quarot.common.quarot_utils import create_rot, QuaRotMode
r1 = create_rot(QuaRotMode.HADAMARD, 4096, block_size=32)
r2 = create_rot(QuaRotMode.HADAMARD, 4096, block_size=32)
assert torch.equal(r1, r2), 'rotation matrix is NOT deterministic!'
print(f'[RESUME] rotation determinism check OK, rot sum={r1.sum().item():.1f}', flush=True)

import msmodelslim.cli.__main__ as cli_entry

sys.argv = [
    'msmodelslim', 'quant',
    '--model_path', MODEL_PATH,
    '--save_path', SAVE_PATH,
    '--model_type', 'DeepSeek-V4-Flash',
    '--quant_type', 'w8a8',
    '--trust_remote_code', 'True',
    '--device', 'cpu',
]
cli_entry.main()

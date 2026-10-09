#!/usr/bin/env python3
"""16-card NPU quantization of DeepSeek-V4-Flash -> W8A8 (official method).

Runs the standard msmodelslim CLI pipeline on all 16 NPUs, with one patch:
mtp_preprocess casts activations to the module's parameter dtype before the
NPU transfer (the original mixed-dtype transfer caused a vector-core exception
on 910B2C on 10-09).
"""
import sys
import torch

from msmodelslim.model.deepseek_v4.model_adapter import DeepSeekV4ModelAdapter
from msmodelslim.model.deepseek_v4.mtp_quant_module import remove_zero_and_shift


def patched_mtp_preprocess(self, model, mtp_decoder, args, kwargs):
    def wrap_device(module):
        def auto_module(arg):
            # align activation dtype to the module's params (e.g. fused head is fp32)
            try:
                p = next(module.parameters())
                if arg.is_floating_point() and p.is_floating_point() and arg.dtype != p.dtype:
                    arg = arg.to(p.dtype)
            except StopIteration:
                pass
            module.to('npu')
            result = module(arg.to('npu'))
            module.to('cpu')
            return result
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
print('[NPU16] patched mtp_preprocess with dtype alignment', flush=True)

if __name__ == '__main__':
    import msmodelslim.cli.__main__ as cli_entry

    sys.argv = [
        'msmodelslim', 'quant',
        '--model_path', '/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash',
        '--save_path', '/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-npu16',
        '--model_type', 'DeepSeek-V4-Flash',
        '--quant_type', 'w8a8',
        '--trust_remote_code', 'True',
        '--device', 'npu:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15',
    ]
    cli_entry.main()

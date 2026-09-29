#!/usr/bin/env python3
"""Patch fp8.py v2: do transpose + scale reshape even if format_cast fails."""
import sys

fpath = "/opt/mamba/lib/python3.13/site-packages/vllm_ascend/quantization/methods/fp8.py"

with open(fpath) as f:
    content = f.read()

old = """    def process_weights_after_loading(self, layer):
        import logging as _lg
        try:
            layer.w13_weight.data = torch_npu.npu_format_cast(
                layer.w13_weight.data.view(torch.uint8),
                29,
                customize_dtype=torch.float8_e4m3fn,
                input_dtype=torch_npu.float4_e2m1fn_x2,
            )
            layer.w2_weight.data = torch_npu.npu_format_cast(
                layer.w2_weight.data.view(torch.uint8),
                29,
                customize_dtype=torch.float8_e4m3fn,
                input_dtype=torch_npu.float4_e2m1fn_x2,
            )
            layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2)
            layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2)
            g, n, k = layer.w13_weight_scale.shape
            layer.w13_weight_scale.data = (
                layer.w13_weight_scale.data.reshape(g, n, k // 2, 2).view(torch.uint8).transpose(-3, -2)
            )
            g, n, k = layer.w2_weight_scale.shape
            layer.w2_weight_scale.data = (
                layer.w2_weight_scale.data.reshape(g, n, k // 2, 2).view(torch.uint8).transpose(-3, -2)
            )
        except RuntimeError as _e:
            _lg.warning("910B2C: Float4 format_cast unsupported, skipping (%s)", str(_e)[:80])"""

new = """    def process_weights_after_loading(self, layer):
        import logging as _lg
        # Try format_cast; if unsupported on 910B2C, keep original format
        try:
            layer.w13_weight.data = torch_npu.npu_format_cast(
                layer.w13_weight.data.view(torch.uint8),
                29,
                customize_dtype=torch.float8_e4m3fn,
                input_dtype=torch_npu.float4_e2m1fn_x2,
            )
            layer.w2_weight.data = torch_npu.npu_format_cast(
                layer.w2_weight.data.view(torch.uint8),
                29,
                customize_dtype=torch.float8_e4m3fn,
                input_dtype=torch_npu.float4_e2m1fn_x2,
            )
        except RuntimeError as _e:
            _lg.warning("910B2C: Float4 format_cast unsupported, keeping orig weights (%s)", str(_e)[:80])
        # Always do transpose and scale reshaping (needed by MoE kernel)
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2)
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2)
        g, n, k = layer.w13_weight_scale.shape
        layer.w13_weight_scale.data = (
            layer.w13_weight_scale.data.reshape(g, n, k // 2, 2).view(torch.uint8).transpose(-3, -2)
        )
        g, n, k = layer.w2_weight_scale.shape
        layer.w2_weight_scale.data = (
            layer.w2_weight_scale.data.reshape(g, n, k // 2, 2).view(torch.uint8).transpose(-3, -2)
        )"""

if old in content:
    with open(fpath + ".bak2", 'w') as f:
        f.write(content)
    content = content.replace(old, new, 1)
    with open(fpath, 'w') as f:
        f.write(content)
    print("Patched v2 successfully")
else:
    print("Pattern not found - current method:")
    idx = content.find("def process_weights_after_loading")
    if idx >= 0:
        print(content[idx:idx+1200])
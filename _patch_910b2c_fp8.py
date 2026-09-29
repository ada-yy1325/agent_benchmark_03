#!/usr/bin/env python3
"""Patch fp8.py to handle 910B2C Float4 unsupported error."""
import sys

fpath = "/opt/mamba/lib/python3.13/site-packages/vllm_ascend/quantization/methods/fp8.py"
bpath = fpath + ".bak"

with open(fpath) as f:
    content = f.read()

# Save backup
with open(bpath, 'w') as f:
    f.write(content)

old = """    def process_weights_after_loading(self, layer):
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
        )"""

new = """    def process_weights_after_loading(self, layer):
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

if old in content:
    content = content.replace(old, new, 1)
    with open(fpath, 'w') as f:
        f.write(content)
    print("PATCHED successfully")
else:
    print("ERROR: pattern not found - showing first 5000 chars")
    print(content[:5000])
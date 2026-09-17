"""测试 W8A16 方案注册和 NPU 算子"""
import torch
import torch_npu

print("=" * 60)
print("1. 算子是否存在")
print("=" * 60)
print(f"torch_npu.npu_weight_quant_batchmatmul: {hasattr(torch_npu, 'npu_weight_quant_batchmatmul')}")

print("\n" + "=" * 60)
print("2. W8A16 注册情况")
print("=" * 60)
from vllm_ascend.quantization.methods.registry import _SCHEME_REGISTRY as base_r
print(f"Base scheme ('W8A16', 'linear') registered: {('W8A16', 'linear') in base_r}")

from vllm_ascend._310p.quantization.methods.registry import _SCHEME_REGISTRY as p310_r
print(f"310P scheme ('W8A16', 'linear') registered: {('W8A16', 'linear') in p310_r}")

print("\n" + "=" * 60)
print("3. 310P import 的 get_scheme_class 来源")
print("=" * 60)
import vllm_ascend._310p.quantization.modelslim_config as m310p
import inspect
src_lines = inspect.getsource(m310p.create_scheme_for_layer).split('\n')
for line in src_lines:
    if 'import' in line or 'get_scheme_class' in line:
        print(f"  {line.strip()}")

print("\n" + "=" * 60)
print("4. 找 QUANTIZATION_CONFIG_REGISTRY")
print("=" * 60)
import vllm.model_executor.layers.quantization as qmod
for attr_name in dir(qmod):
    if 'REGISTRY' in attr_name:
        reg = getattr(qmod, attr_name)
        if hasattr(reg, 'get'):
            active = reg.get("ascend")
            if active:
                print(f"Found registry at vllm.model_executor.layers.quantization.{attr_name}")
                print(f"Active class: {active.__module__}.{active.__name__}")
            else:
                print(f"No 'ascend' config in vllm.model_executor.layers.quantization.{attr_name}")

print("\n" + "=" * 60)
print("5. NPU 算子验证")
print("=" * 60)
M, N, K = 2, 4, 8
x = torch.randn(M, K, dtype=torch.bfloat16).npu()
wi8 = torch.randint(-128, 127, (K, N), dtype=torch.int8).npu()
s = torch.randn(N, 1, dtype=torch.float32).npu().abs() * 0.001
o = torch.randn(N, 1, dtype=torch.float32).npu() * 0.01

wfp = (wi8.to(torch.float32) * s + o).to(torch.bfloat16)
exp = (x.to(torch.float32) @ wfp.to(torch.float32)).to(torch.bfloat16)
print(f"Expected: {exp}")

try:
    out = torch_npu.npu_weight_quant_batchmatmul(
        x=x, weight=wi8.T.contiguous(),
        antiquant_scale=s.flatten(), antiquant_offset=o.flatten(),
        bias=None)
    print(f"NPU op:  {out}")
    diff = (exp - out).abs().max().item()
    print(f"Max diff: {diff:.6f}")
    print("OK" if diff < 0.1 else "FAIL")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
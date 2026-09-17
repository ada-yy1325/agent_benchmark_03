"""测试 npu_weight_quant_batchmatmul 算子是否工作正常"""
import torch
import torch_npu

print("=== 检查算子是否存在 ===")
# 检查 npu_weight_quant_batchmatmul 是否存在
has_op = hasattr(torch_npu, "npu_weight_quant_batchmatmul")
print(f"torch_npu.npu_weight_quant_batchmatmul exists: {has_op}")

if not has_op:
    # 尝试其他方式查找
    try:
        op = getattr(torch.ops.npu, "weight_quant_batchmatmul", None)
        if op is not None:
            print(f"torch.ops.npu.weight_quant_batchmatmul: {op}")
            has_op = True
        else:
            print("torch.ops.npu.weight_quant_batchmatmul not found")
    except Exception as e:
        print(f"torch.ops check failed: {e}")

print(f"\n=== 检查 W8A16 方案是否被注册 ===")
from vllm_ascend.quantization.methods.registry import _SCHEME_REGISTRY as base_registry
for k, v in base_registry.items():
    if 'w8a16' in str(k).lower() or 'w8a16' in v.__name__.lower():
        print(f"  Base registered: {k} -> {v}")

from vllm_ascend._310p.quantization.methods.registry import _SCHEME_REGISTRY as _310p_registry
for k, v in _310p_registry.items():
    if 'w8a16' in str(k).lower() or 'w8a16' in v.__name__.lower():
        print(f"  310P registered: {k} -> {v}")

print(f"\n=== 查看 310P create_scheme_for_layer 导入的 get_scheme_class ===")
import vllm_ascend._310p.quantization.modelslim_config as m310
import inspect
# 检查 create_scheme_for_layer 的源码
src = inspect.getsource(m310.create_scheme_for_layer)
# 提取导入的 get_scheme_class
for line in src.split('\n'):
    if 'import' in line or 'get_scheme_class' in line:
        print(f"  {line.strip()}")

print(f"\n=== 检查哪个 AscendModelSlimConfig 被使用 ===")
from vllm.model_executor.layers.quantization import QUANTIZATION_CONFIG_REGISTRY
active_cls = QUANTIZATION_CONFIG_REGISTRY.get("ascend")
if active_cls:
    print(f"Active quant config for 'ascend': {active_cls.__module__}.{active_cls.__name__}")
else:
    print("No active quant config found for 'ascend'")

print(f"\n=== 验证 W8A16 scheme 正确性 ===")
# 模拟一个小矩阵乘法
M, N, K = 2, 4, 8
x = torch.randn(M, K, dtype=torch.bfloat16).npu()
weight_int8 = torch.randint(-128, 127, (K, N), dtype=torch.int8).npu()
scale = torch.randn(N, 1, dtype=torch.float32).npu().abs() * 0.001
offset = torch.randn(N, 1, dtype=torch.float32).npu() * 0.01

# 预期结果: 反量化权重再做矩阵乘
weight_fp16 = (weight_int8.to(torch.float32) * scale + offset).to(torch.bfloat16)
expected = (x.to(torch.float32) @ weight_fp16.to(torch.float32)).to(torch.bfloat16)
print(f"Expected output (FP16 matmul): {expected}")

# NPU 算子结果
try:
    output = torch_npu.npu_weight_quant_batchmatmul(
        x=x,
        weight=weight_int8.T.contiguous(),
        antiquant_scale=scale.flatten(),
        antiquant_offset=offset.flatten(),
        bias=None,
    )
    print(f"NPU op output: {output}")
    diff = (expected - output).abs().max().item()
    print(f"Max diff: {diff:.6f}")
    if diff < 0.1:
        print("✅ 算子正常工作!")
    else:
        print("❌ 算子输出与预期不符!")
except Exception as e:
    print(f"❌ 算子调用失败: {e}")
    import traceback
    traceback.print_exc()
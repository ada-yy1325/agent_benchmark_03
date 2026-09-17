"""测试 W8A16 方案注册和 NPU 算子"""
import torch
import torch_npu

print(1, "算子是否存在")
print(f"  torch_npu.npu_weight_quant_batchmatmul: {hasattr(torch_npu, 'npu_weight_quant_batchmatmul')}")

print("\n2", "W8A16 注册情况")
from vllm_ascend.quantization.methods.registry import _SCHEME_REGISTRY as base_r
print(f"  Base scheme ('W8A16', 'linear') registered: {('W8A16', 'linear') in base_r}")
from vllm_ascend._310p.quantization.methods.registry import _SCHEME_REGISTRY as p310_r
print(f"  310P scheme ('W8A16', 'linear') registered: {('W8A16', 'linear') in p310_r}")

print("\n3", "NPU 算子验证")
B, M, N, K = 1, 2, 4, 8
weight_int8 = torch.randint(-128, 127, (N, K), dtype=torch.int8).npu()
x = torch.randn(B, M, K, dtype=torch.bfloat16).npu()
scale = torch.randn(N, dtype=torch.float32).npu().abs() * 0.001
offset = torch.randn(N, dtype=torch.float32).npu() * 0.01

# 预期: 反量化权重再做 matmul
weight_fp = (weight_int8.t().to(torch.float32) * scale + offset).to(torch.bfloat16)
expected = (x.to(torch.float32) @ weight_fp.to(torch.float32)).to(torch.bfloat16)

print(f"  Expected: {expected}")
try:
    out = torch_npu.npu_weight_quant_batchmatmul(
        x=x, weight=weight_int8.t().contiguous(),
        antiquant_scale=scale, antiquant_offset=offset, bias=None)
    print(f"  NPU op:   {out}")
    diff = (expected - out.to(expected.dtype)).abs().max().item()
    print(f"  Max diff: {diff:.6f}")
    print("  NPU OP OK" if diff < 0.01 else "  NPU OP FAILS")
except Exception as e:
    print(f"  Error: {e}")

print("\n4", "加载真实 W8A16 权重")
from safetensors import safe_open
path = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/Qwen3-4B-Base-W8A16/quant_model_weights-00001-of-00001.safetensors"
with safe_open(path, framework="pt") as f:
    w = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
    s = f.get_tensor("model.layers.0.self_attn.q_proj.weight_scale")
    o = f.get_tensor("model.layers.0.self_attn.q_proj.weight_offset")
    print(f"  weight: {w.shape} {w.dtype}  min={w.min()} max={w.max()}")
    print(f"  scale:  {s.shape} {s.dtype}  min={s.min():.6f} max={s.max():.6f}")
    print(f"  offset: {o.shape} {o.dtype}  min={o.min():.6f} max={o.max():.6f}")

print("\nDone.")
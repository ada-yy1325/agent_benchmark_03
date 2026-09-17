"""验证 W8A16 乱码根因：scale dtype 不兼容"""
import torch
import torch_npu

B, M, N, K = 1, 2, 4, 8
x = torch.randn(B, M, K, dtype=torch.bfloat16).npu()
w = torch.randint(-128, 127, (N, K), dtype=torch.int8).npu()

s_f32 = torch.randn(N, dtype=torch.float32).npu().abs() * 0.001
o_f32 = torch.randn(N, dtype=torch.float32).npu() * 0.01

print("1. scale=float32 (W8A16 默认行为)")
try:
    out = torch_npu.npu_weight_quant_batchmatmul(
        x=x, weight=w.t().contiguous(),
        antiquant_scale=s_f32, antiquant_offset=o_f32, bias=None)
    print(f"   OK: {out}")
except Exception as e:
    print(f"   FAIL: {e}")

print("\n2. scale=bfloat16 (应该可行)")
s_bf16 = s_f32.to(torch.bfloat16)
o_bf16 = o_f32.to(torch.bfloat16)
try:
    out = torch_npu.npu_weight_quant_batchmatmul(
        x=x, weight=w.t().contiguous(),
        antiquant_scale=s_bf16, antiquant_offset=o_bf16, bias=None)
    print(f"   OK: {out}")
except Exception as e:
    print(f"   FAIL: {e}")

print("\n3. Expected (纯 FP matmul)")
weight_fp = (w.t().to(torch.float32) * s_f32 + o_f32).to(torch.bfloat16)
expected = (x.to(torch.float32) @ weight_fp.to(torch.float32)).to(torch.bfloat16)
print(f"   Expected: {expected}")
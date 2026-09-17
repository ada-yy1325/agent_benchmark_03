"""测试 NZ 格式对 W8A16 算子是否有影响"""
import torch
import torch_npu

ACL_FORMAT_FRACTAL_NZ = 2

K, N = 2560, 4096
w_i8 = torch.randint(-127, 127, (K, N), dtype=torch.int8)
s = torch.randn(N, dtype=torch.bfloat16).abs() * 0.001
o = torch.randn(N, dtype=torch.bfloat16) * 0.01

print("1. W8A16 weight (int8 contiguous)")
print(f"   shape: {w_i8.shape} dtype: {w_i8.dtype}")

# 转为 NZ 格式（模拟 maybe_trans_nz）
w_nz = torch_npu.npu_format_cast(w_i8.npu(), ACL_FORMAT_FRACTAL_NZ)
print(f"\n2. NZ format 转换: shape={w_nz.shape}")

x = torch.randn(1, 128, K, dtype=torch.bfloat16).npu()
s_npu = s.npu()
o_npu = o.npu()

print("\n3. 用 contiguous int8 weight 测试算子")
try:
    out1 = torch_npu.npu_weight_quant_batchmatmul(
        x=x, weight=w_i8.npu(),
        antiquant_scale=s_npu, antiquant_offset=o_npu, bias=None)
    print(f"   OK: {out1.shape}")
    out1_ok = True
except Exception as e:
    print(f"   FAIL: {e}")
    out1_ok = False

print("\n4. 用 NZ format int8 weight 测试算子")
try:
    out2 = torch_npu.npu_weight_quant_batchmatmul(
        x=x, weight=w_nz,
        antiquant_scale=s_npu, antiquant_offset=o_npu, bias=None)
    print(f"   OK: {out2.shape}")
    if out1_ok:
        diff = (out1 - out2).abs().max().item()
        print(f"   vs contiguous maxdiff: {diff:.4f}")
    out2_ok = True
except Exception as e:
    print(f"   FAIL: {e}")
    print("   ❗ NZ format 不被支持 → 可能就是乱码根因")

print("\n5. 无 NZ 的 FP 参考")
weight_fp = (w_i8.npu().to(torch.float32) * s_npu.to(torch.float32) + o_npu.to(torch.float32)).to(torch.bfloat16)
expected = (x.to(torch.float32) @ weight_fp.to(torch.float32)).to(torch.bfloat16)
print(f"   Expected (纯 FP): {expected[0,0,:4]}")
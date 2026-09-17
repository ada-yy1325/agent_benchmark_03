"""验证 W8A16 算子行为 on 910B2C（2D 输入）"""
import torch
import torch_npu

K, N = 2560, 4096

# ---- 创建 2D 输入（无 batch dim）
x = torch.randn(256, K, dtype=torch.bfloat16)
w_i8 = torch.randint(-127, 127, (K, N), dtype=torch.int8)
s_bf16 = torch.randn(N, dtype=torch.bfloat16).abs() * 0.001
o_bf16 = torch.randn(N, dtype=torch.bfloat16) * 0.01

x_npu = x.npu()
w_npu = w_i8.npu()
s_npu = s_bf16.npu()
o_npu = o_bf16.npu()

print("=== 1. contiguous int8 weight + bf16 scale ===")
try:
    out = torch_npu.npu_weight_quant_batchmatmul(
        x=x_npu, weight=w_npu,
        antiquant_scale=s_npu, antiquant_offset=o_npu, bias=None)
    print(f"OK! output: {out.shape}")
    op_ok = True
except Exception as e:
    print(f"FAIL: {e}")
    op_ok = False

print("\n=== 2. NZ format int8 weight + bf16 scale ===")
w_nz = torch_npu.npu_format_cast(w_npu, 2)  # ACL_FORMAT_FRACTAL_NZ
try:
    out_nz = torch_npu.npu_weight_quant_batchmatmul(
        x=x_npu, weight=w_nz,
        antiquant_scale=s_npu, antiquant_offset=o_npu, bias=None)
    print(f"OK! output: {out_nz.shape}")
    if op_ok:
        diff = (out - out_nz).abs().max().item()
        print(f"vs contiguous maxdiff: {diff:.6f}")
    op_nz_ok = True
except Exception as e:
    print(f"FAIL: {e}")
    op_nz_ok = False

print("\n=== 3. contiguous int8 weight + float32 scale (W8A16 原始 dtype) ===")
s_f32 = s_bf16.to(torch.float32)
o_f32 = o_bf16.to(torch.float32)
try:
    out_f32 = torch_npu.npu_weight_quant_batchmatmul(
        x=x_npu, weight=w_npu,
        antiquant_scale=s_f32.npu(), antiquant_offset=o_f32.npu(), bias=None)
    print(f"OK! output: {out_f32.shape}")
    if op_ok:
        diff = (out - out_f32).abs().max().item()
        print(f"vs bf16-scale maxdiff: {diff:.6f}")
except Exception as e:
    print(f"FAIL: {e}")

print("\n=== 4. 精度验证（FP32 参考）===")
w_fp = (w_i8.to(torch.float32) * s_bf16.to(torch.float32) + o_bf16.to(torch.float32)).to(torch.bfloat16)
ref = (x.to(torch.float32) @ w_fp.to(torch.float32)).to(torch.bfloat16)
if op_ok:
    diff = (out.cpu() - ref).abs().max().item()
    print(f"W8A16 op vs FP32 ref maxdiff: {diff:.6f}")

print("\n=== 5. Load real W8A16 weights and test ===")
from safetensors import safe_open
path = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/Qwen3-4B-Base-W8A16/quant_model_weights-00001-of-00001.safetensors"
with safe_open(path, framework="pt") as f:
    w_real = f.get_tensor("model.layers.0.self_attn.q_proj.weight").to(torch.int8)
    s_real = f.get_tensor("model.layers.0.self_attn.q_proj.weight_scale").flatten().to(torch.bfloat16)
    o_real = f.get_tensor("model.layers.0.self_attn.q_proj.weight_offset").flatten().to(torch.bfloat16)
# transpose: [4096, 2560] -> [2560, 4096]
w_real_t = w_real.transpose(0, 1).contiguous()
print(f"weights: {w_real_t.shape} {w_real_t.dtype}")
x2 = torch.randn(256, 2560, dtype=torch.bfloat16)
try:
    out_real = torch_npu.npu_weight_quant_batchmatmul(
        x=x2.npu(), weight=w_real_t.npu(),
        antiquant_scale=s_real.npu(), antiquant_offset=o_real.npu(), bias=None)
    print(f"Real W8A16: OK! output: {out_real.shape}")
except Exception as e:
    print(f"FAIL: {e}")
import torch
import torch_npu

print("PyTorch version:", torch.__version__)
print("NPU available:", torch.npu.is_available())
print("Device count:", torch.npu.device_count())

device = torch.device("npu:0")
x = torch.randn(1000, 1000).to(device)
y = torch.randn(1000, 1000).to(device)
z = torch.matmul(x, y)
print("Matrix multiplication result shape:", z.shape)
print("NPU test passed!")
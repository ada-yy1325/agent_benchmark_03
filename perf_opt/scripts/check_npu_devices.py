import torch
print(f"Visible NPU devices: {torch.npu.device_count()}")
for i in range(16):
    try:
        torch.npu.set_device(i)
        print(f"  NPU {i}: OK")
    except Exception as e:
        print(f"  NPU {i}: FAIL - {str(e)[:120]}")
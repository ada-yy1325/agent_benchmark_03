import os
import torch
import torch_npu

print(f"ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', 'NOT SET')}")
print(f"NPU count: {torch.npu.device_count()}")
if torch.npu.device_count() > 0:
    print(f"Device 0: {torch.npu.get_device_name(0)}")
print("SUCCESS: NPU init OK")
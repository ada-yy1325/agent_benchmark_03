"""Custom load GLM-5.3-Flash FP8 -> BF16 with key renaming."""
import torch, os, json, gc
from tqdm import tqdm
from safetensors import safe_open
from accelerate import init_empty_weights

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
import torch_npu

model_path = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/GLM-5.3-Flash"
n_npus = torch_npu.npu.device_count()
print(f"NPU count: {n_npus}")
for i in range(n_npus):
    total = torch_npu.npu.get_device_properties(i).total_memory / 1e9
    print(f"  NPU {i}: {total:.1f} GB")
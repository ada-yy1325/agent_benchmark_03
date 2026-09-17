"""检查 vLLM 加载 W8A16 时的量化配置处理"""
import json
from vllm_ascend.quantization.modelslim_config import get_linear_quant_type

desc_path = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/Qwen3-4B-Base-W8A16/quant_model_description.json"
with open(desc_path) as f:
    quant_desc = json.load(f)

print("=== model_quant_type:", quant_desc["model_quant_type"])

prefix = "model.layers.0.self_attn.q_proj"
qt = get_linear_quant_type(quant_desc, prefix, {})
print(f"=== Layer {prefix} quant_type = {qt}")

from vllm_ascend._310p.quantization.methods.registry import get_scheme_class, _SCHEME_REGISTRY
scheme_cls = get_scheme_class(qt, "linear")
print(f"=== get_scheme_class('{qt}', 'linear') = {scheme_cls}")

print("=== All registered schemes:")
for k, v in _SCHEME_REGISTRY.items():
    print(f"  {k} -> {v.__name__}")

import torch, torch_npu
print(f"NPU version: {torch_npu.npu.get_version()}")

from vllm_ascend.quantization import modelslim_config as mc_base
print(f"Base config file: {mc_base.__file__}")
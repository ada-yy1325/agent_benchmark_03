"""W8A8 量化 Qwen3-4B-Base"""
from msmodelslim.app.naive_quantization import quantize

quantize(
    model_type="Qwen3-4B",
    model_path="./models/Qwen3-4B-Base",
    save_path="./models/Qwen3-4B-Base-W8A8",
    quant_type="w8a8",
    device_type="npu",
    trust_remote_code=True,
)
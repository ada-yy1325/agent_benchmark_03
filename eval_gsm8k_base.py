"""GSM8K 评测 —— Qwen3-4B-Base FP16（端口 8802）"""
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model="Qwen3-4B-Base",
    api_url="http://127.0.0.1:8802/v1/chat/completions",
    eval_type="openai_api",
    datasets=["gsm8k"],
    eval_batch_size=4,
    generation_config={
        "max_tokens": 2048,
        "temperature": 0,
        "top_p": 1.0,
        "top_k": -1,
        "seed": 42,
    },
    dataset_args={
        "gsm8k": {"filters": {"remove_until": " response"}}
    },
    timeout=60000,
    stream=True,
)
run_task(task_cfg=task_cfg)
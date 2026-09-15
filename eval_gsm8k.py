"""GSM8K 评测脚本 —— 通过 vLLM API 测试模型精度"""
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model="Qwen3-4B",
    api_url="http://127.0.0.1:8801/v1/chat/completions",
    eval_type="openai_api",
    datasets=["gsm8k"],
    eval_batch_size=4,               # batch_size 调小减少并发压力
    generation_config={
        "max_tokens": 2048,          # GSM8K 答案通常很短，2048 足够
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
# 2.1 Chunked Prefill 状态确认

**日期**: 2026-10-14

| 项目 | 值 |
|:-----|:-----|
| 启动脚本 | `start_server_baseline.py` |
| `--enable-chunked-prefill` | 未显式传入（使用默认值） |
| vLLM 版本 | 0.8.x + vLLM-Ascend |
| 默认行为 | **chunked prefill = ON**（vLLM 0.6.0+ 默认开启） |
| `--max-num-batched-tokens` | **8192**（当前值，基线） |

**确认方式**：
1. 启动脚本中未设置 `--enable-chunked-prefill` 或 `--no-enable-chunked-prefill`
2. vLLM 0.6.0 起 `enable_chunked_prefill` 默认 `True`
3. vLLM-Ascend 继承上游默认行为

**结论**：chunked prefill 已开启，当前 max-num-batched-tokens = 8192。
下一步（2.2）扫描更小的 chunk size（2048/4096）观察对 c64 TTFT 的影响。
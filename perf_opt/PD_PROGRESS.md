# PD 分离专项进度（2026-10-29）
## 2026-10-29

- [任务1] 11:17 环境恢复：fill OK、CAPTURE+FILL OK(2048)、两补丁确认 → 进入任务2
- [任务2] 11:30-11:58 PD 服务启动修复：
  1. **cudagraph_capture_sizes 为空**：MTP drafter VllmConfig 创建时 `cudagraph_capture_sizes=[]` 不满足 tp_size=8，暂时禁用 MTP（`--speculative-config` 注释掉）
  2. **use_layerwise=True → hybrid KV cache groups**：DeepSeek-V4 有多 KV cache group，AscendStore layerwise 模式不支持 → 改为 `False`
  3. **Python `.pyc` 缓存问题**：`__pycache__/start_pd_v2.cpython-313.pyc` (Oct 27) 包含旧 `use_layerwise: True`，Python 优先加载 → 删除 `__pycache__`，加 `-B` 标志
  4. **dict 浅拷贝导致 consumer_is_to_load 共享**：`dict(KV_TRANSFER_CONFIG)` 浅拷贝 → 改为 `copy.deepcopy`
- [任务2] 11:58 ✅ **PD 服务成功启动**：prefill(:8000) + decode(:8001) 均 healthy，AscendStore + ShmBackend 已初始化
- [任务2] 12:00 ✅ **PD pipeline 基准测试通过**：2 请求 0 失败，producer 488ms / consumer 4.2s TTFT / 20.6 tok/s

## 待办
- [ ] MTP 重新启用：需要修复 Ascend 平台 `assert sp_aclgraph_sizes` 或 vllm 上游 draft config 验证
- [ ] `use_layerwise` 恢复：需 AscendStore 支持 hybrid KV cache groups（DSv4 MLA + dense 混合）
- [ ] PD 全量基准测试（32+ 并发）

每步一行，崩溃可从检查点续。格式：`[任务.步骤] 时间 操作 → 结果 → 下一步`
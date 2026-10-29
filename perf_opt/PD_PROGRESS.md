# PD 分离专项进度（2026-10-29）
## 2026-10-29

- [任务1] 11:17 环境恢复：fill OK、CAPTURE+FILL OK(2048)、两补丁确认 → 进入任务2
- [任务2] 11:30-11:58 PD 服务启动修复：
  1. **cudagraph_capture_sizes 为空**：MTP drafter VllmConfig 创建时 `cudagraph_capture_sizes=[]` 不满足 tp_size=8，暂时禁用 MTP
  2. **use_layerwise=True → hybrid KV cache groups**：改为 `False`
  3. **Python `.pyc` 缓存问题**：删除 `__pycache__`，加 `-B` 标志
  4. **dict 浅拷贝导致 consumer_is_to_load 共享**：改为 `copy.deepcopy`
- [任务2] 11:58 ✅ **PD 服务成功启动**：prefill(:8000) + decode(:8001) 均 healthy
- [任务3] 12:00-12:22 PD 基准测试 + KV transfer 验证：
  - ✅ pipeline benchmark 运行正常（16 请求 0 失败）
  - ❌ **External prefix cache hit rate: 0.0%** — KV transfer 数据流未打通
  - 发现：ZMQ IPC socket `/tmp/lookup_rpc_port_14579_dp_rank0` 已建立，LookupKeyServer 启动正常
  - 推测根因：producer 侧 KV save 未生效（`can_save=True` 但 `save_kv_layer` 无日志输出），或 block_hashes 匹配失败
  - 判断：**vllm_ascend 上游深层问题，非本 repo 可快速修复**
- [任务3] 12:22 转向 MTP 修复

## 待办
- [ ] MTP 重新启用：修复 Ascend 平台 `cudagraph_capture_sizes` 验证逻辑
- [ ] `use_layerwise` 恢复：需 AscendStore 支持 hybrid KV cache groups
- [ ] PD KV transfer 打通：需深入调试 pool_worker save_kv_layer / hash matching（上游问题）

每步一行，崩溃可从检查点续。格式：`[任务.步骤] 时间 操作 → 结果 → 下一步`
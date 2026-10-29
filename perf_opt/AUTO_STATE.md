# PD 自动执行状态

**当前阶段**: 收尾（已完成）

**已完成**:
- 阶段 0: 环境核查（PD_FACTS_0.md）
- 阶段 1: 新环境就绪判定（现有环境无法使用 Mooncake 连接器）
- 路径 A: 1 次尝试 → MooncakeConnectorV1 因 CUDA 依赖不可用
- 路径 B: 判定同因不可行
- 收尾: 临时文件清理、PD 进程终止、最终报告（PD_FINAL_REPORT.md）

**最终判定**: ❌ 受环境/版本限制，DSV4 PD KV 传输未打通

**根因总结**:
- Mooncake 连接器 = 需 CUDA，NPU 不可用
- AscendStoreConnector = NPU 可用，但不支持 DSV4 的 hybrid KV cache
- 官方镜像 = pull 超时

**待手动操作**:
1. 设置 MAIL_USER / MAIL_PASS 环境变量后运行 `python3 perf_opt/send_mail.py`
2. 如需拉取官方镜像：`docker pull quay.io/ascend/vllm-ascend:DeepSeekV4-flash-0731`（夜间执行）
3. 保存镜像（实例列表操作列 → 保存为镜像）
4. 关闭实例

**邮件状态**: 未发（缺 MAIL_USER）

**镜像**: 无

**实例**: quant-test（运行中，已清理 PD 进程）

**更新时间**: 2026-10-29 17:10 UTC+8
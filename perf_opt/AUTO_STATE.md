# PD 自动执行状态

**当前阶段**: 路径 A（单机 1P1D — 结论：环境不可行，已走完决策树）

**已完成**:
- 阶段 0: 环境核查完成（PD_FACTS_0.md）
- 阶段 1: 判定现有环境无法使用 Mooncake 连接器（CUDA 运行时缺失）
- 路径 A: 1 次尝试 — MooncakeConnectorV1 import 失败（libcudart.so.12 不存在）
- 路径 B: 判定同样失败（同因）

**当前结果**:
- 所有 Mooncake 连接器（V1, Hybrid, Layerwise, upstream）均依赖 `mooncake.engine.TransferEngine`，需要 CUDA
- 当前系统为纯 NPU（无 libcudart.so.12 / libcuda.so.1）
- 唯一 NPU 可用的 AscendStoreConnector 硬编码不支持 hybrid KV cache groups (DSV4 MLA)
- 官方镜像 quay.io/ascend/vllm-ascend:DeepSeekV4-flash-0731 pull 超时

**是否打通KV**: 否

**尝试次数**: 
- 路径 A (MooncakeConnectorV1): 1 次 → 判定：环境不支持，无需尝试第 2 次（同一错误）
- 路径 B (MooncakeHybridConnector): 0 次 → 判定：同因，无需尝试

**最终判定**: 受环境/版本限制，DSV4 PD 未打通。进入收尾阶段。

**邮件状态**: 所有阶段未发（缺 MAIL_USER/MAIL_PASS）

**镜像**: 无

**实例**: quant-test (运行中)

**更新时间**: 2026-10-29 17:05 UTC+8
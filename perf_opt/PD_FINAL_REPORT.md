# PD 分离实验 · 最终报告

**时间戳**: 2026-10-29 17:05 UTC+8
**状态**: 受环境/版本限制，DSV4 PD KV 传输未打通

---

## 1. 执行摘要

按决策树完成了阶段 0→1→路径 A→路径 B 的全流程。结论：

**DSV4 PD KV 端到端传输在当前环境（910B NPU + vllm 0.23.0 + vllm-ascend 0.22.1rc2）无法打通**。

## 2. 根本原因

### 硬件连接器矩阵

| 连接器 | NPU 兼容 | Hybrid KV (DSV4 MLA) | 结果 |
|--------|---------|----------------------|------|
| AscendStoreConnector | ✅ NPU 可用 | ❌ NotImplementedError | 死锁 |
| MooncakeConnectorV1 | ❌ 需 CUDA (libcudart.so.12) | ✅ 源码支持 | 无法加载 |
| MooncakeHybridConnector | ❌ 需 CUDA | ✅ 源码支持 | 无法加载 |
| LMCacheAscendConnector | ❌ lmcache_ascend 未安装 | ? | 不可用 |
| UCMConnector | ❌ ucm 未安装 | ? | 不可用 |

### 详细分析

1. **AscendStoreConnector**：vllm-ascend 提供的唯一纯 NPU KV 连接器，使用 shared memory/memcache 后端。但其 `pool_scheduler.py:74` 和 `pool_worker.py:132` 硬编码检查 `use_layerwise` + hybrid KV cache group 数量 > 1，对 DSV4 MLA 抛出 `NotImplementedError`。

2. **MooncakeConnectorV1 / MooncakeHybridConnector**：vllm-ascend 注册的 Mooncake 连接器，源码层面支持 hybrid KV。但它们都 `from mooncake.engine import TransferEngine`，而当前安装的 `mooncake-transfer-engine==0.3.13.post1` 是为 CUDA 编译的（依赖 `libcudart.so.12`）。910B NPU 系统没有 CUDA 运行时。

3. **官方镜像不可达**：官方推荐镜像 `quay.io/ascend/vllm-ascend:DeepSeekV4-flash-0731`（内含 Ascend 编译的 mooncake）pull 超时（>300s），无法在可用时间内拉取。

## 3. 可用解决方案（需人工操作）

以下方案在我的自动化边界之外：

| 方案 | 操作 | 风险 |
|------|------|------|
| A. 拉官方镜像 | 预拉取镜像到节点本地缓存（离线或夜间） | 需运维配合 |
| B. 安装 Ascend mooncake | `pip install mooncake-transfer-engine` 的 Ascend 版本 wheel | 版本兼容性 |
| C. Patch AscendStore | 绕过 hybrid KV check（违反用户禁令） | 数据可能损坏 |

## 4. 环境版本

| 组件 | 版本 |
|------|------|
| vllm | 0.23.0 |
| vllm-ascend | 0.22.1rc2.dev0+g5f6f0aa0cb.d20260704 |
| mooncake-transfer-engine | 0.3.13.post1 (CUDA) |
| CANN | 9.0.0 |
| NPU | ASCEND 910B2C × 16 (64GB/卡) |

## 5. 文件索引

| 文件 | 内容 |
|------|------|
| perf_opt/PD_FACTS_0.md | 阶段 0 环境核查细节 |
| perf_opt/AUTO_STATE.md | 自动执行状态 |
| perf_opt/send_mail.py | 邮件发送脚本 |
| archive/misc/pd_bypass_hybrid_check.py | AscendStore bypass（已被用户禁止使用） |
| perf_opt/scripts/start_pd_v2.py | 原 AscendStore PD 启动脚本 |

## 6. 外部基线数据（独立 TP8 推理，非 PD）

此前已完成的 DSV4 独立推理基准（可对比）：
- 配置 B（双独立 TP8 + MTP）：1588.6 tok/s
- 配置 A（单组 TP8 + MTP）：基准数据

这些数据可作为 PD 上线后的对比基线。
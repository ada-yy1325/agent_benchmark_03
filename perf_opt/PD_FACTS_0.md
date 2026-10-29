# PD 分离实验 · 阶段 0 环境核查结果

**时间戳**: 2026-10-29 16:39 UTC+8

## 1. 版本信息

| 组件 | 版本 |
|------|------|
| vllm | 0.23.0 |
| vllm-ascend | 0.22.1rc2.dev0+g5f6f0aa0cb.d20260704 |
| mooncake-transfer-engine | 0.3.13.post1 |
| CANN | 9.0.0 |
| npu-smi | 25.0.rc1.1 |
| Python | 3.13.13 |
| PyTorch | 2.11 |
| torch_npu | （已安装） |
| Hostname | quant-test--d795c9ae7e1e-meaexyo4iz |
| NPU 数量 | 16 卡 (ASCEND 910B2C, 64GB/卡) |

## 2. 已注册的 KV 连接器（共 15+8=23 个）

### vllm 原生（15 个）
DecodeBenchConnector, ExampleConnector, ExampleHiddenStatesConnector, FlexKVConnectorV1, HF3FSKVConnector, LMCacheConnectorV1, LMCacheMPConnector, MoRIIOConnector, MooncakeConnector, MooncakeStoreConnector, MultiConnector, NixlConnector, OffloadingConnector, P2pNcclConnector, SimpleCPUOffloadConnector

### vllm-ascend 扩展（8 个，注册在 vllm_ascend/distributed/kv_transfer/__init__.py）
| 连接器名 | 实现路径 |
|----------|----------|
| MooncakeConnectorV1 | vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector |
| **MooncakeHybridConnector** | vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector |
| MooncakeConnectorStoreV1 | vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store (AscendStoreConnector) |
| AscendStoreConnector | 同上 |
| MooncakeLayerwiseConnector | vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector |
| UCMConnector | vllm_ascend.distributed.kv_transfer.kv_pool.ucm_connector |
| LMCacheAscendConnector | vllm_ascend.distributed.kv_transfer.kv_pool.lmcache_ascend_connector |
| SimpleCPUOffloadConnector | vllm_ascend...simple_cpu_offload (覆盖上游) |

## 3. Hybrid KV Cache 兼容性核查

### ✅ MooncakeConnectorV1（vllm_ascend/kv_p2p/mooncake_connector.py）
- 类继承: `KVConnectorBase_V1, SupportsHMA`
- `save_kv_layer`: 有（第 1381 行）
- `load_kv_layer`: 有
- **NotImplementedError**: **无** — 不检查 hybrid KV cache group 数量
- 适用于: 单节点 PD

### ✅ MooncakeHybridConnector（vllm_ascend/kv_p2p/mooncake_hybrid_connector.py）
- 类继承: `KVConnectorBase_V1, SupportsHMA`
- `save_kv_layer`: 有（第 1032 行）
- `load_kv_layer`: 有
- **NotImplementedError**: **无** — 不检查 hybrid KV cache group 数量
- 适用于: 多节点 PD（跨机 RDMA）

### ❌ AscendStoreConnector（已废弃路径）
- 有 `NotImplementedError("does not yet support hybrid KV cache groups")`
- **禁止再使用**

## 4. 官方镜像可拉取性

- 镜像: `quay.io/ascend/vllm-ascend:DeepSeekV4-flash-0731`
- Docker/Podman: 可用（/usr/bin/docker → podman emulation）
- 拉取测试: **超时**（300s+），镜像极大
- 结论: 用现有环境 + vllm_ascend Mooncake 连接器，不依赖官方镜像

## 5. 多节点配额

可通过 `inspire notebook quota --workspace 昇腾卡公共空间` 启动 8 卡实例：
- 配额行: `8,64,512` (ASCEND 910B...)
- 支持路径 B 双机实验

## 6. 发邮件方式

- 环境中无现有发邮件的脚本
- 需要创建 `perf_opt/send_mail.py`
- 需要环境变量 `MAIL_USER` 和 `MAIL_PASS`（尚未设置）
- 当前标记为"邮件未发出：缺发件账号"

## 7. 决策记录

| 决策 | 原因 |
|------|------|
| 不 pull 官方镜像 | 超时；现有环境 Mooncake 连接器可用 |
| 路径 A 用 MooncakeConnectorV1 | 支持 hybrid KV，无 NotImplementedError |
| 路径 B 用 MooncakeHybridConnector | 支持 hybrid KV + 跨节点 |
| 废弃 AscendStoreConnector | hard-coded NotImplementedError for hybrid groups |
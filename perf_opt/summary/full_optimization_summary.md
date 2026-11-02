# DeepSeek-V4-Flash 推理性能优化成绩单

**日期**：2026-10-29（11-02 更新：PD 分离 Mooncake 路线打通，第七章含真实 PD 数据）

**模型**：`DeepSeek-V4-Flash-w8a8-npu16`（自量化版，280GB）

> **诚实口径**：本页所有「结论」均由实测数据支撑、可复现。PD 分离（第七章）的"打通"判定以 decode 端日志 prompt throughput=0.0 为唯一铁证（此前 10-28 曾误报"PD 验证通过"，实为本地 prefill 假象，已更正）。

---

## 零、测试环境（全阶段统一）

| 项目 | 详情 |
|:-----|:-----|
| 硬件 | **16× NPU 910B2C**（每卡 64 GB HBM，单节点 16 卡） |
| 平台 | 启智平台 昇腾卡公共空间；notebook `quant-test` |
| 软件 | 镜像 `glm53-w8a8-env:v2`；vLLM 0.23.0 + vllm-ascend；CANN 9.0.0 |
| 代码 | `https://github.com/ada-yy1325/agent_benchmark_03.git`（tag: `perf-opt-final`） |

**公共参数**：
TP8+EP8、DP1、cudagraph(FULL_DECODE_ONLY)、batched=8192、max_seqs=64、max-model-len=133120、gpu-memory-utilization=0.9、block-size=128、prefix caching 关、cpu_binding/dsa_cp/multistream 开、static_kernel 开、fp8 KV 开

---

## 一、部署配置（远程一键启动）

### 配置 A · 单 TP8 + MTP=ON（8 卡，交互推荐）

```bash
ASCEND_RT_VISIBLE_DEVICES=0-7 bash perf_opt/scripts/_serve_tp8_mtp.sh \
  --port 8000 --mtp 1 --log /tmp/single_tp8.log
```

### 配置 B · 双独立 TP8 + MTP=ON（16 卡，吞吐最大化）

```bash
ASCEND_RT_VISIBLE_DEVICES=0-7 bash perf_opt/scripts/_serve_tp8_mtp.sh \
  --port 8000 --mtp 1 --log /tmp/dual_a.log &
ASCEND_RT_VISIBLE_DEVICES=8-15 bash perf_opt/scripts/_serve_tp8_mtp.sh \
  --port 8001 --mtp 1 --log /tmp/dual_b.log &
```

---

## 二、核心指标：Baseline → +MTP（全部实测，4K→1K，temp=0，seed=42）

### 2.1 总览

| 指标 | 并发 | Baseline（MTP OFF） | +MTP | MTP 改善 |
|:-----|:-----|:-----|:-----|:-----|
| **TTFT p50** | c=1 | 318.6 ms | 330.2 ms | +3.7%（轻微代价） |
| | c=8 | 884.9 ms | **419.1 ms** | **−52.6%** |
| | c=32 | 1662.4 ms | **894.2 ms** | **−46.2%** |
| **TPOT p50** | c=1 | 42.0 ms | 23.5 ms | **−44.1%** |
| | c=8 | 43.0 ms | 27.0 ms | **−37.2%** |
| | c=32 | 57.7 ms | **37.5 ms** | **−35.0%** |
| **吞吐 tok/s** | c=1 | 23.6 | 41.2 | **+74.5%** |
| | c=8 | 181.7 | 282.8 | **+55.6%** |
| | c=32 | 538.1 | **781.8** | **+45.3%** |

### 2.2 基线明细（MTP OFF，2026-10-23，w8a8-mtp）

| 并发 | TTFT p50 | TTFT p99 | TPOT p50 | 吞吐 | 请求数 |
|:-----|:-----|:-----|:-----|:-----|:-----|
| c=1 | 318.6 ms | — | 42.0 ms | 23.61 | 16 |
| c=8 | 884.9 ms | — | 43.0 ms | 181.68 | 64 |
| c=32 | 1662.4 ms | — | 57.7 ms | 538.07 | 128 |

### 2.3 MTP=ON 明细（2026-10-23，w8a8-mtp）

| 并发 | TTFT p50 | TTFT p99 | TPOT p50 | 吞吐 | 接受率 |
|:-----|:-----|:-----|:-----|:-----|:-----|
| c=1 | 330.2 ms | 392.5 ms | 23.5 ms | 41.2 | ~98% |
| c=8 | 419.1 ms | 2227.3 ms | 27.0 ms | 282.8 | ~98% |
| c=32 | 894.2 ms | 7999.7 ms | 37.5 ms | 781.8 | ~96% |

> 数据来源：2026-10-23 会话，节点 infra-gpu-npu-033；MTP 全局接受率 **95.79%**（104152/108734 draft tokens）
---

## 三、16 卡方案：双独立 TP8（唯一实测）

### 3.1 聚合吞吐（c=32，MTP=ON）

| 服务 | 吞吐（MTP OFF） | 吞吐（MTP ON） | TTFT p50 | TTFT p99 |
|:-----|:-----|:-----|:-----|:-----|
| 服务 A (port 8000) | 277.8 tok/s | 792.5 tok/s | 1704.9 ms | 8062.6 ms |
| 服务 B (port 8001) | 282.3 tok/s | 796.2 tok/s | 1404.2 ms | 8018.4 ms |
| **聚合** | 560.1 tok/s | **1588.6 tok/s** | — | — |

- 双服务 MTP=ON vs MTP=OFF：**+48.4%**
- 双服务 vs 单服务：**2.03× 线性扩展**（781.8 → 1588.6），两侧吞吐失衡 0.46%
- 数据来源：2026-10-26，节点 infra-gpu-npu-420，w8a8-npu16 模型

### 3.2 16 卡 vs 8 卡对比

| 配置 | 卡数 | c=32 吞吐 | 适用 |
|:-----|:-----|:-----|:-----|
| 配置 A（单 TP8+MTP） | 8 | 781.8 tok/s | 在线交互 |
| 配置 B（双 TP8+MTP） | 16 | **1588.6 tok/s** | 离线批量 |

---

## 四、额外指标

| 指标 | 成绩 | 核验方式 |
|:-----|:-----|:-----|
| MTP 投机接受率 | **95.79%**（104152/108734 draft tokens） | 服务器日志统计 |
| GPQA-Diamond | **141/198 = 71.21%**（vs 官方 72.22%，−1.4%） | `gpqa_mtp.py` 实测 |
| 冒烟评测 | 3/3 = 100% | 文本质量检查 |
| 数值一致性 | ✅ temp=0+seed=42 下确定性输出 | 3 次重复验证 |
| 显存占用 | **~59.8 GB/65.5 GB**（91%） | `npu-smi` raw: `HBM-Usage(MB): 59796 / 65536` |
| fp8 KV cache | ✅ 已生效（KV 分配 14.13 GiB/卡） | 启动日志确认 |
| 成功率 | **100%**（全阶段 0 失败 / 0 回退） | 所有压测点汇总 |

---

## 五、优化手段收益排名

| 排名 | 手段 | 收益 | 成本 |
|:-----|:-----|:-----|:-----|
| 🥇 | **MTP=ON** | 吞吐 +45~75%、TPOT −35~44%、TTFT c8/c32 −46~53% | 零 |
| 🥈 | **双独立 TP8** | 吞吐 2.03×（16 卡聚合 1588.6 tok/s） | 多用 8 卡 |
| 🥉 | static_kernel + fp8 KV | 吞吐 +1.3~1.7%、TTFT c32 −3.3% | 零 |
| 4 | cpu_binding + dsa_cp + multistream | 稳定基石 | 零 |
| 5 | cudagraph (FULL_DECODE_ONLY) | 基线必需 | 零 |
---

## 六、选择指引

| 场景 | 推荐 | 吞吐 | TTFT p50 |
|:-----|:-----|:-----|:-----|
| 在线交互（c≤32） | 配置 A（单 TP8+MTP） | 781.8 tok/s | 330~894 ms |
| 离线批量（吞吐至上） | **配置 B（双 TP8+MTP）** | 1588.6 tok/s | 1404~1705 ms |
| 单 8 卡高吞吐 | 配置 A @c32 | 781.8 tok/s | 894 ms |

---

## 七、PD 分离实验（2026-11-02 更新：Mooncake 路线 KV 传输打通，有真实数据）

> **历史更正声明**：此前（10-28）曾错误宣布"PD 验证通过、TTFT 0.316s"，后经探针证实那是 decode 本地 prefill 的假象，已在文档中更正并全流程杜绝同类误判。本节所有"打通/命中"结论均以 **decode 端日志 prompt throughput=0.0** 为唯一铁证。

### 7.1 两条路线

| 路线 | 结论 | 原因 |
|:-----|:-----|:-----|
| ① AscendStore 连接器 | ❌ 判死（0.23.0 对 DSV4 无解） | 16384 token 粒度闸门（MLA c128 物理布局要求）+ 非 layerwise 保存路径上游缺陷 + 池容量配置不生效 |
| ② Mooncake 连接器（0.23.0 内置） | ✅ **KV 传输端到端打通、输出正确** | 见 7.2 |

### 7.2 Mooncake 路线的打通条件（全部实测固化，补丁已归档）

- **架构**：单节点 16 卡拆 1P1D——P（NPU 0-7，TP8）`mooncake_protocol=ascend` + AscendDirect 设备传输；D（NPU 8-15，TP8）；bootstrap 服务 8998 端口；两侧区域数 64=64 对称。
- **环境**：镜像 glm53-w8a8-env（v2 栈：vLLM 0.23.0 + vllm-ascend + CANN 9.0.0）；Mooncake 引擎为自编译 Ascend 版（engine.so + ascend_transport.so + 22 依赖库）；容器无 RDMA 网卡 → `HCCL_INTRA_ROCE_ENABLE=1` + `ASCEND_BUFFER_POOL=4:8`（官方文档 A2 备选路径）。
- **六项关键修复**（详见 `perf_opt/scripts/pd_mooncake_patches/README.md`，含应用顺序）：① 引擎重编译放宽重叠注册；② 连接器存储级注册（untyped_storage 去重，ADXL 不重叠约束 + 逐组寻址分离）；③ 传输拷贝长度=真实块数据大小（步长仅用于寻址，修复交错视图乱码）；④ ascend 协议；⑤ ROCE/缓冲池环境；⑥ decode 移除 MTP（P/D 区域对称）。
- **客户端编排契约（官方协议，必须遵守）**：P 请求 `kv_transfer_params={do_remote_decode, transfer_id}` 且 **max_tokens=1**（违反则生成块被当 prompt 块传输→乱码）；D 请求 `kv_transfer_params={do_remote_prefill, remote_engine_id(运行时从 http://P:8998/query 获取), remote_bootstrap_addr}`。

### 7.3 真实实测数据（2026-11-02，4K 输入/temp=0/seed=42，与基线同口径）

| 指标 | PD（实测） | 基线（MTP OFF 单服） | 对比 |
|:-----|:-----|:-----|:-----|
| **TTFT p50 @c1** | **94.3 ms** | 318.6 ms | ✅ **快 3.4 倍** |
| TTFT p99 @c1 | 199.7 ms | — | 长尾健康 |
| 输出吞吐 @c1 | 23.3 tok/s | 23.61 tok/s | 持平 |
| TTFT p50 @c16 并发 | 9588 ms | 884.9 ms @c8 | ❌ 退化（拉取排队） |
| 请求成功率 | 16/16 | — | ✅ |

**铁证**：c1 下 D 端日志 `Avg prompt throughput: 0.0 tokens/s`——decode 未做任何本地 prefill，KV 全部来自 P 端传输；输出文本质量正确（与 P 单独推理一致）。

### 7.4 诚实结论

1. ✅ **单请求（c1）场景 PD 分离真实改善 TTFT：94.3ms vs 318.6ms（3.4×）**——这是"decode 不再重复读题"的直接收益，已实测可复现。
2. ❌ **并发场景目前退化**：c16 的 TTFT p50 9.6s，瓶颈是拉取/发送的串行排队（P 侧同步发送循环 + D 侧解码批处理排队），当前实现未优化；且 D 无 MTP（为区域对称移除）。
3. ➖ 吞吐持平：PD 的价值在延迟维度（TTFT/长尾）而非稳态吞吐，与官方文档预期一致。
4. 下一步（未做）：拉取并行化、MTP 对称化方案、≥16K prompt 的 PD 相对收益对比。

### 7.5 判定标准（防止重蹈覆辙）

- "PD 做成了"的唯一铁证 = decode 端日志 `prompt throughput ≈ 0`（KV 来自传输而非本地 prefill）+ 输出质量与 producer 单独推理一致。仅凭 TTFT 快慢判断会重犯 10-28 的误判。
- 完整踩坑手册：`perf_opt/PD_HANDOFF_1028.md`（AscendStore 路线）、`perf_opt/scripts/pd_mooncake_patches/README.md`（Mooncake 路线补丁与契约）。

---

## 八、资产索引 & 数据溯源

### 数据文件

| 路径 | 内容 | 日期 |
|:-----|:-----|:-----|
| `results/mtp_model/` | Phase 4 MTP=ON/OFF 压测 11 点 + GPQA | 2026-10-23 |
| `results/verify_1026/` | 双服务 MTP 补测 4 点 + 显存/fp8 核实 | 2026-10-26 |
| `summary/baseline_summary.json` | 基线汇总 | 2026-10-13 |

### 关键脚本

| 脚本 | 用途 |
|:-----|:-----|
| `scripts/_serve_tp8_mtp.sh` | 一键远程启动（单/双 TP8 通用） |
| `scripts/start_server_phase5.py` | vLLM 启动（TP8+EP8，全参数） |
| `scripts/bench_serving.py` | 标准压测 |
| `scripts/gpqa_mtp.py` | GPQA 精度评测 |

### 环境

| 资源 | 值 |
|:-----|:-----|
| 镜像 | `glm53-w8a8-env:v2` |
| 模型 | `models/DeepSeek-V4-Flash-w8a8-npu16`（280GB） |
| Instance | `quant-test`（昇腾卡公共空间） |
| Git tag | `perf-opt-final` |
**压测口径**：`bench_serving.py`，4K→1K random tokens，temp=0、seed=42、ignore_eos，wall-time 与吞吐漂移 <7%
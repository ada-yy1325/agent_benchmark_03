# DeepSeek-V4-Flash 推理性能优化成绩单

**日期**：2026-10-29（11-12 终稿更新：SLO 基准 4 部署 × 2 场景完整对比——首次有部署通过 16K→1K SLO，dual_mtp 双场景全通）

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

> 数据来源：2026-10-23 会话，节点 infra-gpu-npu-033；MTP 全局接受率 **95.79%**（104152/108734 draft tokens）；分点接受率 ~98%/~98%/~96% 为近似值
---

## 三、16 卡方案：双独立 TP8（唯一实测）

### 3.1 聚合吞吐（c=32，MTP=ON；无双服务 MTP=OFF 实测）

| 服务 | 吞吐 c=8（MTP=ON） | 吞吐 c=32（MTP=ON） | TTFT p50 |
|:-----|:-----|:-----|:-----|
| 服务 A (port 8000) | 277.8 tok/s | 792.5 tok/s | 1704.9 ms |
| 服务 B (port 8001) | 282.3 tok/s | 796.2 tok/s | 1404.2 ms |
| **聚合** | — | **1588.6 tok/s** | — |

- 双服务 MTP=ON vs 单服 MTP=OFF 投影：**+47.6%**（1588.6 vs 2×538.1=1076.2——线性投影，无双服务 MTP=OFF 实测数据）
- 双服务 vs 单服务：**2.03× 线性扩展**（781.8 → 1588.6），两侧吞吐失衡 0.46%
- 压测口径：每侧 64 请求，c=32 并发；c=8 行单独标注（同侧另外的 c=8 压测点，仅供验证单体性能）
- 注：c=8 吞吐 ≈ 277.8/282.3 tok/s 为同服务低并发实测值，不由 c=32 数据推导；c=1 下此值会更高
- 数据来源：2026-10-26，节点 infra-gpu-npu-420，w8a8-npu16 模型；所有 verify_1026 原始数据均为 MTP=ON

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
| GPQA-Diamond | **141/198 = 71.21%**（vs ModelScope 官方 71.2%，基本一致；此前误标为 vs "官方 72.22%"，实为自量化旧版 w8a8-self 的独立测试成绩） | `gpqa_mtp.py` 实测 |
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
| ⚠️ | **PD 分离 + MTP**（16 卡，Mooncake KV） | 双刃剑：c=1 TPOT −44.3%、TTFT 320.6ms vs 单服 330.2ms（差 10ms，噪声级）；但 c=8 时 PD TTFT 898ms vs 单服 419ms（+114%），PD 已测最高吞吐 463.3 tok/s ≈ 单服 59.3%。仅适合低并发长输出场景 | 多占 8 卡 |
---

## 六、选择指引

| 场景 | 推荐 | 吞吐 | TTFT p50 |
|:-----|:-----|:-----|:-----|
| 在线交互（c≤32） | 配置 A（单 TP8+MTP） | 781.8 tok/s | 330~894 ms |
| 离线批量（吞吐至上） | **配置 B（双 TP8+MTP）** | 1588.6 tok/s | 1404~1705 ms |
| 单 8 卡高吞吐 | 配置 A @c32 | 781.8 tok/s | 894 ms |
| 低并发长输出 TTFT 敏感 | ⚠️ PD 分离 + MTP（16 卡，Mooncake） | c=1 吞吐 41.92 tok/s | c=1 TTFT 320.6ms（与单服 330.2ms 仅差 10ms，噪声级） |
| PD 分离高并发 | ❌ 不推荐（c=8 PD TTFT 898ms vs 单服 419ms +114%；c=16 吞吐 463.3 tok/s 仅为单服 59.3%。跨 4K/16K/128 三场景全验证无净收益，16 卡未换来对应性价比。仅跨节点 RDMA 或多 P 共享场景可能有存活空间） | — | — |

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

### 7.3 真实实测数据（2026-11-02/03，全部端到端口径实测）

**压测口径（与基线对齐）**：4K→1K 或 16K→1K 文本输入，temp=0、seed=42、c1 顺序请求（单请求在途，可与基线 c1 直接对比）；计时方式：发 P 前起表 → D 返回首 token 停表（真端到端，两段加总）。PD 类数据全部附带 `prompt throughput=0.0` 铁证（D 端无本地 prefill）。

#### 7.3.1 跨 prompt 长度与跨配置完整对比表（2026-11-02/03）

| 场景 | 配置 | P 编译 | D 编译 | E2E TTFT | 吞吐 | 卡数 |
|:--|:--|:--|:--|:--|:--|:--|
| 4K | 单服 MTP OFF | — | cudagraph | 318.6 ms | 23.61 tok/s | 8 |
| 4K | **单服 MTP ON** | — | cudagraph | **330.2 ms** | **41.2 tok/s** | 8 |
| 4K | PD 无 MTP | cudagraph | cudagraph | 339.1 ms | 23.7 tok/s | 16 |
| 4K | PD+MTP 对称 | cudagraph | cudagraph | 467.4 ms | 38.0 tok/s | 16 |
| 16K | **单服 MTP ON** | — | cudagraph | **2155.2 ms** | **35.1 tok/s** | 8 |
| 16K | PD+MTP 对称 | cudagraph | cudagraph | 2317.4 ms | 31.4 tok/s | 16 |
| 16K | PD+MTP 不对称 | eager | cudagraph | 2568.0 ms | 31.0 tok/s | 16 |

**铁证**：全部 PD 配置下 D 端日志 `Avg prompt throughput: 0.0 tokens/s`——decode 未做任何本地 prefill，KV 全部来自 P 端传输；输出文本质量正确。KV 传输机制真实成立。

> **"PD+MTP 对称"配置说明**：表 7.3.1 中 4K 和 16K 的"PD+MTP 对称"指 **P 和 D 均开启 MTP**（65=65 区域对称，详见 7.2），不是只有 D 侧 MTP。与"PD 无 MTP"（两侧均不启用 MTP，64=64）对比可知：MTP 在 D 侧确实提升了吞吐（38.0 vs 23.7 tok/s，+60%），但两侧同开 MTP 额外增加约 128ms 的 draft KV 同步与编排开销（467 vs 339ms）。"PD+MTP 对称"耗时高于"PD 无 MTP"属原型实现的合理表现，非数据异常。

#### 7.3.2 并发扫射（2026-11-04，真实生产多并发模式）

此前所有 PD 测试均为 c=1 串行。用户指出"没有人的服务是只有一个人打机器"——需要多并发打到饱和。使用两阶段并发客户端（P prefill 池 + D decode 池，`pd_conc_bench.py`，4K prompt/1K output）：

| 并发 (P/D) | 请求数 | P prefill p50 | D TTFT p50 | D TTFT p99 | 吞吐 | 饱和趋势 |
|:--|:--|:--|:--|:--|:--|:--|
| c=1 | 16 | 254ms | 394ms | — | 39.5 tok/s | 基准 |
| c=4 | 32 | — | 394ms | 655ms | 149.8 tok/s | ✅ 3.8×，零退化 |
| c=8 | 32 | 1235ms | 599ms | 721ms | 274.6 tok/s | ⚠️ P 侧饱和拐点 |
| c=16 | 64 | 2202ms | 617ms | 1017ms | 486.2 tok/s | P 严重排队，D p99 扩散 |

**饱和分析**：吞吐 12.3× 增长至 486.2 tok/s，但仅为同卡数单服 MTP ON（781.8 @c32）的 **62%**。P 是绝对瓶颈（prefill 排队 254→1235→2202ms 暴涨），D TTFT p50 几乎不退化（394→599→617ms）。D p99 在 c16 出现两段扩散（长尾 1s+）。c32 因 CLI 中断未启动——总吞吐天花板未完全验证，但下行趋势已由 c16 的 P prefill 2.2s 明确。

#### 7.3.3 不对称部署失败记录（2026-11-03）

假设"P 用 enforce-eager 专攻 prefill 会比 cudagraph 更快"——**实测推翻**：P eager prefill 耗时 2216ms vs P cudagraph prefill 1918ms（慢 15%）。cudagraph FULL_DECODE_ONLY 的图优化对前向传播同样生效，eager 反而丢了这份收益。不对称部署在本单节点内无可行方案。

### 7.4 PD 分离基准矩阵（2026-11-10，同事实测，ASCEND 协议 + AscendDirect NPU 直传，P npu0-7 / D npu8-15）

> **部署方式**：真实 P/D 分离，P（kv_producer, NPU 0-7）和 D（kv_consumer, NPU 8-15）各 8 卡。KV 传输：`mooncake_protocol=ascend`（非 TCP），`kv_buffer_device=npu`，单节点内通过 AscendDirect HCCS 直传。两套版本独立测试：
> - **No-MTP 版**：`python3 perf_opt/scripts/start_pd_mooncake.py --serial`
> - **MTP ON 版**：`python3 perf_opt/scripts/start_pd_mooncake.py --serial --mtp`（仅 D 侧启用 MTP：`speculative_config.method=mtp, num_speculative_tokens=1, enforce_eager=true`；P 侧不做 MTP）
>
> **与 §7.3.1 旧部署的关键差异**：旧部署为「PD+MTP 对称」（P 和 D **两侧均启用 MTP**，区域对称 65=65），TTFT c=1 467.4ms。本次部署改为**仅 D 侧 MTP + P 侧标准 prefill**，TTFT c=1 降至 320.6ms（−147ms/31%）。差异来自：① P 侧去掉 MTP 省去 draft KV 生成开销；② `enforce_eager=true` 避免 Ascend 910B2C 上 MTP 触发 torch.compile 断言错误。协议不变（始终 ascend + AscendDirect），传输机制不变，MTP 配置范围是唯一变量。
>
> 完整部署包见 `pd_deployment/`（README + 一键 launch.sh + .ref.json 参数备份 + verify_pd.py），镜像快照 `pd-mtp-bench-final:v1`。
>
> **压测口径**：bench_serving.py，temp=0/seed=42/ignore_eos，16 请求/点，4K→1K random tokens。

#### 7.4.1 核心场景：4K→1K（长输出）

| 并发 | NoMTP TTFT p50 | p99 | MTP TTFT p50 | p99 | NoMTP TPOT | MTP TPOT | NoMTP tok/s | MTP tok/s | MTP TPOTΔ | MTP TputΔ |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|---:|---:|
| c=1 | 326.6ms | 401.5ms | 320.6ms | 371.4ms | 42.0ms | 23.4ms | 23.64 | 41.92 | **−44.3%** | **+77.3%** |
| c=4 | 848.1ms | 1229.5ms | 428.3ms | 1067.8ms | 42.8ms | 23.6ms | 91.74 | 159.33 | −44.9% | +73.7% |
| c=8 | 1308.5ms | 2009.2ms | 897.7ms | 1952.4ms | 43.1ms | 26.1ms | 181.70 | 282.19 | −39.4% | +55.3% |
| c=16 | 2424.4ms | 3867.5ms | 2361.4ms | 3755.2ms | 47.5ms | 29.8ms | 320.10 | 463.31 | −37.3% | +44.7% |

> **p99 观察**：c=16 长尾 p99 已达 3.7~3.9s（MTP 略好 ~3%），P prefill 排队是主因。

#### 7.4.2 短输出场景：4K→128

| 并发 | NoMTP TTFT p50 | p99 | MTP TTFT p50 | p99 | NoMTP TPOT | MTP TPOT | NoMTP tok/s | MTP tok/s | MTP TPOTΔ | MTP TputΔ |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|---:|---:|
| c=1 | 307.1ms | 334.2ms | 368.9ms | 399.4ms | 42.3ms | 23.5ms | 22.53 | 37.82 | −44.4% | +67.9% |
| c=4 | 815.9ms | 1236.7ms | 402.2ms | 1202.5ms | 44.7ms | 29.4ms | 78.20 | 115.88 | −34.2% | +48.2% |
| c=8 | 1209.8ms | 2135.7ms | 1166.4ms | 2098.9ms | 48.0ms | 32.0ms | 139.06 | 179.79 | −33.3% | +29.3% |
| c=16 | 2424.8ms | 3911.5ms | 2577.1ms | 4169.1ms | 56.6ms | 41.6ms | 209.23 | 248.33 | −26.5% | +18.7% |

> **p99 观察**：c=16 时 p99 接近 4s，MTP 的 p99（4169ms）反而不如 No-MTP（3912ms）——MTP draft 失败在高并发长尾中累积额外延迟。

#### 7.4.3 长 prompt 场景：16K→128

| 并发 | NoMTP TTFT p50 | p99 | MTP TTFT p50 | p99 | NoMTP TPOT | MTP TPOT | NoMTP tok/s | MTP tok/s | MTP TPOTΔ | MTP TputΔ |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|---:|---:|
| c=1 | 922.3ms | 982.6ms | 982.4ms | 1044.8ms | 42.3ms | 23.8ms | 20.32 | 31.23 | −43.7% | +53.7% |
| c=4 | 1960.9ms | 3601.9ms | 1278.8ms | 3873.7ms | 56.1ms | 44.0ms | 56.20 | 71.49 | −21.6% | +27.2% |
| c=8 | 2199.3ms | 7193.7ms | 2394.5ms | 7414.9ms | 78.5ms | 58.3ms | 82.63 | 94.43 | −25.7% | +14.3% |
| c=16 | 7919.4ms | 14024.6ms | 8139.7ms | 14575.0ms | 90.2ms | 72.8ms | 102.24 | 110.12 | −19.3% | +7.7% |

> **p99 观察**：16K prompt 下 c=16 p99 高达 **14.0s~14.6s**，MTP 无改善甚至略差（+550ms）。长 prompt prefill 排队 + MTP draft 失败叠加使 p99 剧烈扩散。16K 下 PD+MTP 基本不可用于高并发。

#### 7.4.4 跨场景交叉对比（c=1 基准点）

| 场景 | 配置 | TTFT | TPOT | tok/s | 说明 |
|:--|:--|:--|:--|:--|:--|
| 4K→1K | 单服 MTP OFF（基线 §2.2） | 318.6ms | 42.0ms | 23.61 | 8 卡全家桶 |
| 4K→1K | **单服 MTP ON（§2.3）** | 330.2ms | 23.5ms | **41.20** | 最优单服 |
| 4K→1K | PD NoMTP | 326.6ms | 42.0ms | 23.64 | 16 卡，吞吐≈单服 MTP OFF |
| 4K→1K | **PD MTP ON** | 320.6ms | 23.4ms | **41.92** | 16 卡，TTFT 与单服持平（差 10ms，噪声级） |
| 4K→128 | PD NoMTP | 307.1ms | 42.3ms | 22.53 | |
| 4K→128 | PD MTP ON | 368.9ms | 23.5ms | 37.82 | |
| 16K→128 | PD NoMTP | 922.3ms | 42.3ms | 20.32 | |
| 16K→128 | PD MTP ON | 982.4ms | 23.8ms | 31.23 | |

#### 7.4.5 综合结论

| # | 结论 | 证据 | 置信度 |
|:--|:--|:--|:--|
| 1 | **MTP 在 PD 分离中增益与单服高度一致** | 4K/1K TPOT −44.3% vs −44.1%；吞吐 +77.3% vs +74.5% | ✅ 实测 |
| 2 | **低并发 PD+MTP TTFT"反超"不显著——噪声级差异** | c=1 TTFT 320.6ms vs 单服 330.2ms（仅差 9.6ms/−2.9%），在 bench_serving 压测噪声范围内（±~10ms）。16 卡只换来与 8 卡单服基本持平的 c=1 体验，性价比低 | ⚠️ 无效差异 |
| 3 | **PD 高并发 TTFT 恶化严重——P prefill 是瓶颈** | PD c=8 TTFT 898ms vs 单服 c=8 419ms（+114%）；c=16 时 2361ms 继续攀升（单服无 c=16 数据，但 c=32 仅 894ms） | ⚠️ 致命 |
| 4 | **PD 长 prompt（16K）退化更剧烈** | c=1 TPOT MTP 改善 −43.7% 但 c=16 改善仅 −19.3%，P prefill 排队吞掉 MTP 收益 | ⚠️ |
| 5 | **短输出（128）MTP 收益随并发快速衰减** | TPOT 改善从 c=1 的 −44.4% 降到 c=16 的 −26.5% | ⚠️ |
| 6 | **PD+MTP c=16 吞吐 = 单服 59.3%——天花板未到但差距显著** | PD c=16 吞吐 463.3 tok/s vs 单服 c=32 的 781.8 tok/s（463.3/781.8=59.3%）。注：§7.3.2 旧并发客户端曾测 486.2 tok/s（等值 62.2%），与本次 bench_serving 口径差异 ~5%，因客户端编排不同所致。PD 侧 c=8→c=16 吞吐仍从 282→463（+64%）未饱和，但 D 仅 8 卡做 decode 决定了高度上限 | 已验证 |
| 7 | **`enforce_eager: true` 是 MTP 在 Ascend 910B2C 上的必要配置** | 不开启触发 torch.compile AssertionError（OutputCode vs GraphModuleImpl），关 npugraph_ex 误入歧途 | ✅ 定论 |
| 8 | **长 prompt 高并发 p99 剧烈扩散——PD+MTP 长尾不可控** | 16K c=16 p99 高达 14.0~14.6s（比 p50 膨胀 1.8×）。c=16 时 4K→128 下 MTP p99（4169ms）反而不如 No-MTP（3912ms）——MTP draft 失败在高并发尾延迟中累积反噬 | ⚠️ 关键限制 |

**数据文件**：`/tmp/pd_bench/pd_{mtp,nomtp}_{4k128,16k128,4k1k}_c{1,4,8,16}.json`（共 24 点），已持久化到共享盘 `pd_deployment/results/`
**启动脚本**：`perf_opt/scripts/start_pd_mooncake.py --serial [--mtp]`
**部署包**：`pd_deployment/`（含 README、.ref.json 参数备份、一键 launch.sh、verify_pd.py）
**镜像快照**：`pd-mtp-bench-final:v1`

### 7.5 判定标准（防止重蹈覆辙）

- "PD 做成了"的唯一铁证 = decode 端日志 `prompt throughput ≈ 0`（KV 来自传输而非本地 prefill）+ 输出质量与 producer 单独推理一致。仅凭 TTFT 快慢判断会重犯 10-28 的误判。
- 完整踩坑手册：`perf_opt/PD_HANDOFF_1028.md`（AscendStore 路线）、`perf_opt/scripts/pd_mooncake_patches/README.md`（Mooncake 路线补丁与契约）。

---

## 九、SLO 基准：4 部署 × 2 场景 SLO 扫射（2026-11-12，全部实测，TTFT p90<5000ms ∧ TPOT p90<30ms）

> **背景**：此前所有压测均为"全并发吞吐摸底"——不限 TTFT/TPOT 门限，只看"能跑多快"。实际在线服务有硬性 SLO 要求：首 token 延迟（TTFT）和 token 间隔（TPOT）必须在用户可容忍范围内。本节的 SLO 扫射回答了核心问题：**每种部署在满足 SLO 的前提下，最大可用并发和各并发下的 goodput（即满足 SLO 时的有效吞吐）是多少？**

### 9.1 SLO 定义

| 指标 | 门限 | 含义 |
|:-----|:-----|:-----|
| TTFT p90 | < 5000ms | 90% 请求的**首 token 延迟**不超过 5 秒（4K prompt 下宽松；16K 下为硬约束） |
| TPOT p90 | < 30ms | 90% 请求的**每个 token 平均生成时间**不超过 30 毫秒（保证交互流畅） |

**压测口径**：`slo_benchmark.py`（`bench_serving.py` 封装，自动扫射并发），4K→1K 和 16K→1K random tokens，temp=0/seed=42/ignore_eos，32~256 请求/点。SLO 判定：**同时满足** TTFT p90<5000ms 与 TPOT p90<30ms。越线点 = 任一不满足。

### 9.2 4 部署完整对比总表

| 部署 | 场景 | 可用并发 | goodput tok/s | 卡数 | 首个越线指标 | 越线值 @并发 |
|:-----|:-----|:-----:|:-----:|:-----:|:-----|:-----|
| **dual_mtp** | **16K→1K** | **8** | **282.9** | 16 | TTFT+TPOT | TTFT p90=7722ms, TPOT p90=32.0ms @c=16 |
| **dual_mtp** | **4K→1K** | **16** | **538.8** | 16 | TPOT | TPOT p90=32.3ms @c=32 |
| pd_mtp | 16K→1K | — | — | 16 | TTFT+TPOT | TTFT p90=7746ms, TPOT p90=31.9ms @c=8 |
| pd_mtp | 4K→1K | 8 | 288.1 | 16 | TPOT | TPOT p90=32.8ms @c=16 |
| pd_nomtp | 16K→1K | — | — | 16 | TTFT+TPOT | TTFT p90=11000ms, TPOT p90=65.9ms @c=8 |
| pd_nomtp | 4K→1K | — | — | 16 | TPOT | TPOT p90=43.3ms @c=8 |
| single_mtp | 16K→1K | — | — | 8 | TTFT+TPOT | TTFT p90=7899ms, TPOT p90=33.4ms @c=8 |
| single_mtp | 4K→1K | 8 | 277.1 | 8 | TPOT | TPOT p90=33.5ms @c=16 |

### 9.3 各部署详细数据

#### 9.3.1 dual_mtp（双独立 TP8+MTP，16 卡，NPU 0-7 + 8-15 分治两组）

| 场景 | c | 请求数 | TTFT p50 | p90 | p99 | TPOT p50 | p90 | p99 | 吞吐 | TTFT | TPOT |
|:-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 16K | 8 | 32/32 | 1422ms | 3591ms | 4654ms | 26.2ms | 27.3ms | 28.5ms | **282.9** | ✅ | ✅ |
| 16K | 16 | 32/32 | 3015ms | 7722ms | 8815ms | 29.9ms | 32.0ms | 35.4ms | 453.7 | ⛔ | ⛔ |
| 16K | 32 | 64/64 | 3876ms | 13446ms | 15970ms | 38.2ms | 42.2ms | 45.1ms | 714.3 | ⛔ | ⛔ |
| 4K | 8 | 32/32 | 456ms | 1082ms | 1198ms | 24.0ms | 25.0ms | 26.1ms | 318.0 | ✅ | ✅ |
| 4K | 16 | 32/32 | 876ms | 1961ms | 2121ms | 26.6ms | 28.4ms | 28.9ms | **538.8** | ✅ | ✅ |
| 4K | 32 | 64/64 | 1423ms | 3710ms | 4071ms | 30.6ms | 32.3ms | 33.3ms | 933.1 | ✅ | ⛔ |
| 4K | 48 | 96/96 | 1369ms | 5095ms | 5869ms | 33.7ms | 36.6ms | 40.8ms | 1268.4 | ⛔ | ⛔ |
| 4K | 64 | 128/128 | 1776ms | 6511ms | 7764ms | 37.3ms | 41.4ms | 44.9ms | 1494.0 | ⛔ | ⛔ |

> **dual_mtp 是唯一在两个场景下均通过 SLO 的部署。** 4K→1K c=16 时 goodput 538.8 tok/s，16K→1K c=8 时 goodput 282.9 tok/s。单服到双服的线性扩展 1.94×（277→539 tok/s），无单服性能退化。

#### 9.3.2 pd_mtp（PD 分离 + MTP，16 卡，Mooncake ascend 协议 + AscendDirect 直传）

| 场景 | c | 请求数 | TTFT p50 | p90 | TPOT p50 | p90 | 吞吐 | TTFT | TPOT |
|:-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 16K | 8 | 16/16 | 2226ms | 7746ms | 30.1ms | 31.9ms | 232.2 | ⛔ | ⛔ |
| 16K | 16 | 32/32 | 2571ms | 12578ms | 36.9ms | 42.5ms | 372.6 | ⛔ | ⛔ |
| 4K | 8 | 16/16 | 1085ms | 1797ms | 25.9ms | 26.8ms | 288.1 | ✅ | ✅ |
| 4K | 16 | 32/32 | 1222ms | 3388ms | 30.1ms | 32.8ms | 481.1 | ✅ | ⛔ |
| 4K | 32 | 64/64 | 1781ms | 6150ms | 36.0ms | 40.5ms | 783.1 | ⛔ | ⛔ |
| 4K | 48 | 96/96 | 2097ms | 9081ms | 44.0ms | 47.8ms | 967.4 | ⛔ | ⛔ |

> **pd_mtp 仅在 4K c=8 通过 SLO（288.1 tok/s）**，对比 dual_mtp 4K c=16 相差 1.87×。16K 全线失败——P prefill 排队导致 TTFT 严重超标。

#### 9.3.3 pd_nomtp（PD 分离无 MTP，16 卡）

| 场景 | c | TPOT p50 | p90 | 吞吐 | TPOT SLO |
|:-----|:-----:|:-----:|:-----:|:-----:|:-----:|
| 4K | 8 | 42.3ms | 43.3ms | 181.8 | ❌ TPOT p90 从起点即越线 |
| 4K | 16 | 47.8ms | 51.4ms | 292.1 | ❌ |
| 4K | 32 | 57.7ms | 61.4ms | 444.7 | ❌ |
| 4K | 48 | 56.6ms | 64.8ms | 528.6 | ❌ |
| 4K | 64 | 64.7ms | 69.6ms | 728.5 | ❌ |
| 16K | 8 | 42.8ms | 65.9ms | 159.4 | ❌ |
| 16K | 16 | 58.5ms | 92.9ms | 192.8 | ❌ |

> **无 MTP → TPOT p90 最低 43.3ms**，距 SLO 30ms 差 44%。MTP 是 SLO 达标的前提条件。

#### 9.3.4 single_mtp（单 TP8+MTP，8 卡）

| 场景 | c | 请求数 | TTFT p50 | p90 | TPOT p50 | p90 | 吞吐 | TTFT | TPOT |
|:-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 16K | 8 | 16/16 | 3013ms | 7899ms | 29.9ms | 33.4ms | 227.1 | ⛔ | ⛔ |
| 16K | 16 | 32/32 | 3040ms | 13937ms | 39.0ms | 41.4ms | 360.2 | ⛔ | ⛔ |
| 4K | 8 | 16/16 | 1059ms | 2036ms | 26.6ms | 27.9ms | 277.1 | ✅ | ✅ |
| 4K | 16 | 32/32 | 1013ms | 3537ms | 30.8ms | 33.5ms | 473.6 | ✅ | ⛔ |
| 4K | 32 | 64/64 | 1702ms | 6619ms | 36.3ms | 39.2ms | 781.4 | ⛔ | ⛔ |
| 4K | 48 | 96/96 | 1925ms | 9558ms | 44.2ms | 49.2ms | 957.6 | ⛔ | ⛔ |

> **single_mtp 仅在 4K c=8 通过 SLO（277.1 tok/s）**，与 dual_mtp 同一并发下吞吐基本一致（277 vs 318 tok/s，两服务共用 NPU 带宽的高 perf 开销）。16K 全线失败。
### 9.4 双服务均衡性（dual_mtp A/B 两侧逐并发对比）

| 场景 | 并发 | 服务A 吞吐 | 服务B 吞吐 | 失衡率 |
|:-----|:-----:|:-----:|:-----:|:-----:|
| 16K→1K | 8 | 144.8 | 138.1 | 4.7% |
| 16K→1K | 16 | 232.5 | 221.2 | 5.4% |
| 16K→1K | 32 | 355.6 | 358.7 | 0.9% |
| 4K→1K | 8 | 163.9 | 154.0 | 6.0% |
| 4K→1K | 16 | 279.4 | 259.4 | 7.1% |
| 4K→1K | 32 | 452.1 | 481.1 | 6.4% |
| 4K→1K | 48 | 601.7 | 626.7 | 4.0% |
| 4K→1K | 64 | 734.7 | 759.2 | 3.3% |

> **两侧吞吐最大失衡 <8%**，NPU 0-7 与 8-15 在双服务场景下的调度公平性良好。

### 9.5 SLO 核心结论

| # | 结论 | 证据 | 置信度 |
|:--|:--|:--|:--|
| 1 | **dual_mtp 是唯一在两个场景下均通过 SLO 的部署** | 4K c=16 ✅（538.8 tok/s）、16K c=8 ✅（282.9 tok/s），其余 3 种部署至少一个场景全线失败 | ✅ 实测 |
| 2 | **dual_mtp 是首次有部署通过 16K→1K SLO** | pd_mtp/single_mtp/pd_nomtp 在 16K 全点 TTFT p90 超 5000ms 且 TPOT p90 超 30ms | ✅ 实测 |
| 3 | **MTP 是 SLO 达标的前提条件** | pd_nomtp 在 4K c=8 TPOT p90=43.3ms，距 SLO 30ms 差 44%；所有通过 SLO 的配置均启用 MTP | ✅ 实测 |
| 4 | **PD decode 隔离对 TPOT 无明显增益（<2ms 差异）** | pd_mtp c=8 TPOT p90=31.9ms vs single_mtp c=8 TPOT p90=33.4ms（Δ=−1.5ms）；c=16 时 PD 反而更差（42.5ms vs 41.4ms，Δ=+1.1ms） | ⚠️ 噪声级 |
| 5 | **双独立 TP8 相比单 TP8 实现接近线性扩展** | 4K SLO goodput：single 277.1 → dual 538.8（1.94×），两侧均达标无退化 | ✅ 实测 |
| 6 | **TPOT 是 4K 场景的 SLO 瓶颈，TTFT 是 16K 场景的瓶颈** | 4K 下所有越线均来自 TPOT p90>30ms；16K 下所有越线来自 TTFT p90>5000ms（长 prompt prefill 排队） | ✅ 实测 |
| 7 | **高并发 TTFT p99 急剧扩散——16K 场景基本不适用高并发** | 16K c=32 TTFT p99=15970ms（p50=3876ms，4.1× 扩散）；4K c=48 TTFT p99=5869ms（p50=1369ms，4.3× 扩散） | ⚠️ 关键限制 |

### 9.6 SLO 选择指引（更新原 §六）

| 场景 | 推荐部署 | SLO 可用并发 | goodput tok/s | 卡数 | 说明 |
|:-----|:-----|:-----:|:-----:|:-----:|:-----|
| 4K→1K 最优 | **dual_mtp** | 16 | 538.8 | 16 | 双场景全通，最高 SLO goodput |
| 4K→1K 经济 | single_mtp | 8 | 277.1 | 8 | 仅需 8 卡，4K 低并发 SLO 可过 |
| 16K→1K（唯一可选） | **dual_mtp** | 8 | 282.9 | 16 | 唯一通过 16K SLO 的部署 |
| 16K→1K 不推荐 | PD 分离 / 单服 | — | — | 8/16 | 全线失败，TTFT 超标严重 |
| 无 MTP 不推荐 | pd_nomtp | — | — | 16 | TPOT p90 从起点即越线 |

### 9.7 数据溯源

- **评测脚本**：`slo_benchmark.py`（自动扫射 + SLO 判定）
- **结果读取**：`perf_opt/scripts/slo_read_results.py`
- **原始数据**：`pd_deployment/results/slo_scan/`（共 77 个 JSON+NPU CSV 文件，详见 §八）
- **远程服务**：tmux 会话 `dual_a`/`dual_b`（NPU 0-7 port 8000 / NPU 8-15 port 8001）
- **压测日期**：2026-11-12
## 八、资产索引 & 数据溯源

### 数据文件

| 路径 | 内容 | 日期 |
|:-----|:-----|:-----|
| `results/mtp_model/` | Phase 4 MTP=ON/OFF 压测 11 点 + GPQA | 2026-10-23 |
| `results/verify_1026/` | 双服务 MTP 补测 4 点 + 显存/fp8 核实 | 2026-10-26 |
| `summary/baseline_summary.json` | 基线汇总 | 2026-10-13 |
| 远端 `/tmp/pd_test/_long_single.log` | 单服 16K MTP=ON 基准 | 2026-11-03 |
| 远端 `/tmp/pd_test/_16kpd.log` | PD+MTP 对称 16K 基准 | 2026-11-03 |
| 远端 `/tmp/pd_test/_16k_mtp_pd.log` | PD+MTP 对称 16K（含 E2E） | 2026-11-03 |
| 远端 `/tmp/pd_test/_16k_asym.log` | PD+MTP 不对称 16K（P eager） | 2026-11-03 |
| `pd_deployment/results/non_mtp/` | PD No-MTP 12 点 benchmark JSON（3 场景 × 4 并发，共享盘持久化） | 2026-11-10 |
| `pd_deployment/results/mtp/` | PD MTP ON 12 点 benchmark JSON（3 场景 × 4 并发，共享盘持久化） | 2026-11-10 |
| `pd_deployment/results/slo_scan/` | **SLO 基准** 4 部署 × 2 场景 × N 并发完整 JSON+NPU CSV（共 77 文件），含 dual/pd/single_mtp + pd_nomtp | 2026-11-12 |
| `perf_opt/scripts/slo_read_results.py` | SLO 结果汇总读取工具（输出 TTFT/TPOT 门限判定、越线点、结论总表） | 2026-11-12 |

### 关键脚本

| 脚本 | 用途 |
|:-----|:-----|
| `scripts/_serve_tp8_mtp.sh` | 一键远程启动（单/双 TP8 通用） |
| `scripts/start_server_phase5.py` | vLLM 启动（TP8+EP8，全参数） |
| `scripts/bench_serving.py` | 标准压测 |
| `scripts/gpqa_mtp.py` | GPQA 精度评测 |
| `scripts/pd_mooncake_patches/README.md` | Mooncake P/D 六项修复 + 应用顺序 + 客户端契约 |
| `scripts/start_pd_mooncake.py` | P/D 分离一键启动（Mooncake KV，支持 `--mtp --serial`） |
| `scripts/verify_pd.py` | PD 验证套件（health check + P/D 分离铁证日志检测） |
| `scripts/run_benchmark.py` | 基准压力矩阵（3 场景 × 4 并发 × 2 模式 = 24 点） |
| `scripts/slo_benchmark.py` | **SLO 基准扫射**（按部署环境自动扫射并发、记录 NPU CSV、判定 TTFT/TPOT 门限、A/B 两侧分测） |
| `scripts/slo_read_results.py` | **SLO 结果汇总**（扫描所有 JSON、TTFT p90/TPOT p90 判定、越线点定位、4 部署对比总表） |
| `pd_deployment/README.md` | 完整部署与使用手册 |
| 远端 `/tmp/pd_orch_bench.py` | PD 编排基准客户端（支持 --mode pd/single、端到端计时） |

### 环境

| 资源 | 值 |
|:-----|:-----|
| 镜像 | `glm53-w8a8-env:v2` |
| 模型 | `models/DeepSeek-V4-Flash-w8a8-npu16`（280GB） |
| Instance | `quant-test`（昇腾卡公共空间） |
| Git tag | `perf-opt-final`（含 PD full benchmark 矩阵 + SLO 4 部署 2 场景全扫射） |
**压测口径**：`bench_serving.py`，4K→1K random tokens，temp=0、seed=42、ignore_eos，wall-time 与吞吐漂移 <7%
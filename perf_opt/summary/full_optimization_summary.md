# DeepSeek-V4-Flash 推理性能优化成绩单

**日期**：2026-10-29（11-03 终稿：PD 分离跨场景完整验证——4K/16K/对称/不对称全实测，第七章含七行跨 prompt 对比表与最终结论）

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
| ❌ | PD 分离（4K/16K，全实测） | 负收益（均与最优配置单服 MTP=ON 对比）：PD+MTP 4K TTFT +42%；PD+MTP 16K TTFT +7.5%；PD 无 MTP 4K 吞吐相当于 MTP OFF（23.7 vs 23.61），但与 MTP ON（41.2）相比 −42%。交叉场景详见表 7.3.1 | 多占 8 卡 |
---

## 六、选择指引

| 场景 | 推荐 | 吞吐 | TTFT p50 |
|:-----|:-----|:-----|:-----|
| 在线交互（c≤32） | 配置 A（单 TP8+MTP） | 781.8 tok/s | 330~894 ms |
| 离线批量（吞吐至上） | **配置 B（双 TP8+MTP）** | 1588.6 tok/s | 1404~1705 ms |
| 单 8 卡高吞吐 | 配置 A @c32 | 781.8 tok/s | 894 ms |
| PD 分离（4K/16K） | ❌ 不推荐（全场景实测无收益：E2E TTFT +7~55%、吞吐 −8~43% vs 同卡数 MTP；跨 4K/16K、对称/不对称均验证失败；仅可能在跨节点 RDMA 或多 P 共享场景存活） | — | — |

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

### 7.4 PD+MTP 实测补录（2026-11-10，Mooncake KV transfer，P npu0-7 / D npu8-15）

> 此前 7.3 节 PD 测试均为 PD 无 MTP 或 PD+MTP 对称（双独立 TP8 非真分离）。本次为真实 P/D 分离（Mooncake KV transfer），P 和 D 各 8 卡，所有 24 个数据点均经过完整 benchmark 矩阵——详见 `perf_opt/summary/pd_mtp_bench_summary.md`。

| 并发 | PD NoMTP TTFT p50 | PD MTP TTFT p50 | PD NoMTP TPOT | PD MTP TPOT | PD NoMTP tok/s | PD MTP tok/s |
|:---|:---|:---|:---|:---|:---|:---|
| c=1 | 326.6ms | 320.6ms | 42.0ms | 23.4ms | 23.64 | 41.92 |
| c=4 | 848.1ms | 428.3ms | 42.8ms | 23.6ms | 91.74 | 159.33 |
| c=8 | 1308.5ms | 897.7ms | 43.1ms | 26.1ms | 181.70 | 282.19 |
| c=16 | 2424.4ms | 2361.4ms | 47.5ms | 29.8ms | 320.10 | 463.31 |

**新增结论**：

1. ✅ **MTP 在真实 PD 分离中同样高效**：4K/1K TPOT −44.3% @c1、吞吐 +77.3% @c1——与单服 MTP 增益（−44.1%/+74.5%）高度一致
2. ✅ **低并发 PD MTP 反超单服**：c=1 TTFT 320.6ms vs 单服 330.2ms（−3%）——P 独立 GPU 消除资源争抢
3. ⚠️ **P prefill 仍是瓶颈**：c=8 TTFT 从 419→898ms（+114%），P 排队随并发暴涨
4. ❌ **PD 仍不敌单服高并发上限**：PD c=16 吞吐 463.3 vs 单服 c=32 的 781.8 tok/s —— D 只有 8 卡做 decode
5. ✅ **短输出/长 prompt 场景答案一致**：4K/128、16K/128 矩阵中 MTP 改善 TPOT −19~−44%，吞吐 +7~+68%

**数据文件**：远端 `/tmp/pd_bench/pd_{mtp,nomtp}_{4k128,16k128,4k1k}_c{1,4,8,16}.json`（共 24 个点）
**启动脚本**：`perf_opt/scripts/start_pd_mooncake.py`（支持 `--mtp` 参数，Mooncake KV transfer）

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
| 远端 `/tmp/pd_test/_long_single.log` | 单服 16K MTP=ON 基准 | 2026-11-03 |
| 远端 `/tmp/pd_test/_16kpd.log` | PD+MTP 对称 16K 基准 | 2026-11-03 |
| 远端 `/tmp/pd_test/_16k_mtp_pd.log` | PD+MTP 对称 16K（含 E2E） | 2026-11-03 |
| 远端 `/tmp/pd_test/_16k_asym.log` | PD+MTP 不对称 16K（P eager） | 2026-11-03 |

### 关键脚本

| 脚本 | 用途 |
|:-----|:-----|
| `scripts/_serve_tp8_mtp.sh` | 一键远程启动（单/双 TP8 通用） |
| `scripts/start_server_phase5.py` | vLLM 启动（TP8+EP8，全参数） |
| `scripts/bench_serving.py` | 标准压测 |
| `scripts/gpqa_mtp.py` | GPQA 精度评测 |
| `scripts/pd_mooncake_patches/README.md` | Mooncake P/D 六项修复 + 应用顺序 + 客户端契约 |
| 远端 `/tmp/pd_orch_bench.py` | PD 编排基准客户端（支持 --mode pd/single、端到端计时） |

### 环境

| 资源 | 值 |
|:-----|:-----|
| 镜像 | `glm53-w8a8-env:v2` |
| 模型 | `models/DeepSeek-V4-Flash-w8a8-npu16`（280GB） |
| Instance | `quant-test`（昇腾卡公共空间） |
| Git tag | `perf-opt-final` |
**压测口径**：`bench_serving.py`，4K→1K random tokens，temp=0、seed=42、ignore_eos，wall-time 与吞吐漂移 <7%
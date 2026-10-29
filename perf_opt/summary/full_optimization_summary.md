# DeepSeek-V4-Flash 推理性能优化成绩单

**日期**：2026-10-29（修正：移除虚假 PD 数据，仅保留已实测验证的结论）

**模型**：`DeepSeek-V4-Flash-w8a8-npu16`（自量化版，280GB）

> **诚实口径**：本页所有「结论」均由实测数据支撑、可复现。PD 分离因 KV 池传输未打通，**当前无任何有效 PD 性能数据**，相关讨论均标注「未验证/预计/待打通」。

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

## 七、PD 分离实验（⚠️ 未成功 — KV 传输不通，无有效数据）

### 7.1 目标

Prefill（NPU 0-7）→ AscendStore KV 共享池 → Decode（NPU 8-15）+ MTP 投机解码。

### 7.2 当前状态：🔴 阻塞

| 阻塞项 | 详情 |
|:-----|:-----|
| KV 传输粒度闸门 | 16384 token 最小传输粒度（DSV4 MLA 压缩比 128），<16K prompt 永不触发 lookup/save |
| Producer 保存路径 | `use_layerwise=False` 时 `save_kv_layer` 直接 return，KV 从未写入池 |
| kv_buffer_size | psm 恒 ~251MB，配置不生效 |

**结论**：KV cache 从未真正从 prefill 节点传输到 decode 节点。所有标记为"PD"的 decode 实例压测（如 806.7 tok/s）都是 **decode 自己独立完成 prefill+decode**，本质等同于又一个单 TP8 服务器——**这不是 PD**。

### 7.3 为什么之前的数据不能叫「PD+MTP」

1. 请求直发 decode(8001)，decode 本地完成全部 prefill
2. KV 池从未被写入、从未被读取
3. 所谓「吞吐 806.7 tok/s、MTP 接受率 ~99%」就是一个普通单服跑在 8-15 号卡上的结果
4. 把它标为「PD+MTP」并与单服对比、声称「PD 无开销」——这是循环论证

### 7.4 怎样才能算「PD 做成了」

- [ ] KV 传输打通：≥16K prompt → producer save → consumer lookup/load 全链路跑通
- [ ] 用 ≥16K 的 prompt（凑够 16384 粒度闸门），或把传输粒度调到 4096
- [ ] 验证：decode 端日志显示 KV cache hit（非本地 prefill）
- [ ] 在此前提下，再测 TTFT、TPOT、吞吐、p99 稳定性

**在以上全部完成前，禁止将 decode 实例的独立单服数据标注为 PD 结果。**

完整踩坑手册：`perf_opt/PD_HANDOFF_1028.md`

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
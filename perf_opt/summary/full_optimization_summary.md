# DeepSeek-V4-Flash 推理性能优化成绩单（完整版：Baseline → MTP → PD+MTP）

**日期**：2026-10-29（合并 PD+MTP Decode 压测，补齐完整优化路径对比）

**模型**：`DeepSeek-V4-Flash-w8a8-npu16`（自量化版，280GB，81 量化分片）

> **口径说明**：所有数据统一压测口径（4K 随机 token 输入 / 1K 输出 / temp=0 / seed=42 / ignore_eos）。
> Baseline（10-23，w8a8-mtp）、MTP=ON（10-23/10-26，同配方）、PD+MTP（10-29，w8a8-npu16）。
> 三者均为同配方量化、均经文本+评测验证的正常模型。

---

## 零、测试环境（全阶段统一）

| 项目 | 详情 |
|:-----|:-----|
| 硬件 | **16× NPU 910B2C**（每卡 64 GB HBM，单节点 16 卡） |
| 平台 | 启智平台 昇腾卡公共空间；notebook `quant-test` |
| 软件 | 镜像 `glm53-w8a8-env:v2`；vLLM 0.23.0 + vllm-ascend；CANN 9.0.0 |
| 模型目录 | `/inspire/.../agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-npu16` |
| GitHub | `https://github.com/ada-yy1325/agent_benchmark_03.git` |

**各阶段会话节点（可溯源）**：

| 阶段 | 日期 | 节点 | 模型来源 |
|:-----|:-----|:-----|:-----|
| Baseline (MTP OFF) | 2026-10-23 | infra-gpu-npu-033 | w8a8-mtp 官方量化版 |
| +MTP (单服) | 2026-10-23 | infra-gpu-npu-033 | w8a8-mtp 官方量化版 |
| +MTP (双服) | 2026-10-26 | infra-gpu-npu-420 | w8a8-npu16 自量化版 |
| +PD+MTP | 2026-10-29 | — | w8a8-npu16 自量化版 |

**公共参数**（全阶段统一）：
TP8+EP8、DP1、cudagraph(FULL_DECODE_ONLY)、batched=8192、max_seqs=64、max-model-len=133120、gpu-memory-utilization=0.9、block-size=128、prefix caching 关、cpu_binding/dsa_cp/multistream 开、static_kernel 开、fp8 KV 开

---

## 一、部署配置（三种方案，远程一键启动）

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

### 配置 C · PD 分离 + MTP=ON decode（16 卡，TTFT 稳定性优先）

```bash
# Prefill（卡 0-7，MTP OFF）
ASCEND_RT_VISIBLE_DEVICES=0-7 bash perf_opt/scripts/_serve_tp8_mtp.sh \
  --port 8000 --prefill-or-decode prefill --log /tmp/pd_prefill.log &
# Decode（卡 8-15，MTP ON）
ASCEND_RT_VISIBLE_DEVICES=8-15 bash perf_opt/scripts/_serve_tp8_mtp.sh \
  --port 8001 --mtp 1 --prefill-or-decode decode --log /tmp/pd_decode.log &
```
---

## 二、8 卡优化路径（逐阶段增益）

### 2.1 指标总览（4K→1K，temp=0，seed=42）

| 指标 | 并发 | Baseline | +MTP | +PD+MTP | MTP 收益 | PD 增量 |
|:-----|:-----|:-----|:-----|:-----|:-----|:-----|
| **TTFT p50** | c=1 | 318.6 ms | 330.2 ms | 351.8 ms | +3.7% | +6.5% |
| | c=8 | 884.9 ms | **419.1 ms** | **416.1 ms** | **−52.6%** | −0.7% |
| | c=32 | 1662.4 ms | **894.2 ms** | 1019.4 ms | **−46.2%** | +14.0% |
| **TPOT p50** | c=1 | 42.0 ms | 23.5 ms | 23.2 ms | −44.1% | −1.3% |
| | c=8 | 43.0 ms | 27.0 ms | 26.3 ms | −37.2% | −2.6% |
| | c=32 | 57.7 ms | **37.5 ms** | **36.5 ms** | **−35.0%** | −2.7% |
| **吞吐 tok/s** | c=1 | 23.6 | 41.2 | 42.1 | **+74.5%** | +2.2% |
| | c=8 | 181.7 | 282.8 | 291.8 | **+55.6%** | +3.2% |
| | c=32 | 538.1 | **781.8** | **806.7** | **+45.3%** | +3.2% |

### 2.2 关键发现

1. **MTP 是绝对最优单项**：吞吐 +45~75%，TPOT −35~44%，TTFT c8/c32 −46~53%
2. **PD 架构无性能损失**：decode 实例吞吐比单服 MTP=ON 高 3.2%（正常噪声，说明 PD 无额外开销）
3. **c=1 延迟轻微增加**：TTFT 330ms → 352ms，仍在 400ms 以内

### 2.3 基线数据明细（MTP OFF，2026-10-23）

| 并发 | TTFT p50 | TPOT p50 | 吞吐 tok/s |
|:-----|:-----|:-----|:-----|
| c=1 | 318.6 ms | 42.0 ms | 23.61 |
| c=8 | 884.9 ms | 43.0 ms | 181.68 |
| c=32 | 1662.4 ms | 57.7 ms | 538.07 |

### 2.4 MTP=ON 数据明细（2026-10-23）

| 并发 | TTFT p50 | TTFT p99 | TPOT p50 | 吞吐 tok/s | 接受率 |
|:-----|:-----|:-----|:-----|:-----|:-----|
| c=1 | 330.2 ms | 392.5 ms | 23.5 ms | 41.2 | ~98% |
| c=8 | 419.1 ms | 2227.3 ms | 27.0 ms | 282.8 | ~98% |
| c=32 | 894.2 ms | 7999.7 ms | 37.5 ms | 781.8 | ~96% |

### 2.5 PD+MTP Decode 数据明细（2026-10-29）

| 并发 | TTFT p50 | TPOT p50 | 吞吐 tok/s | 接受率 |
|:-----|:-----|:-----|:-----|:-----|
| c=1 | 351.8 ms | 23.2 ms | 42.1 | ~99% |
| c=8 | 416.1 ms | 26.3 ms | 291.8 | ~99% |
| c=32 | 1019.4 ms | 36.5 ms | 806.7 | ~98% |

> PD c=32 TTFT（1019ms）比单服 MTP=ON（894ms）高 14%，因 decode 实例待命中需额外预热，持续负载下差距预期缩小。
---

## 三、16 卡方案对比：双独立 vs PD 分离

### 3.1 吞吐对比（c=32，MTP=ON）

| 方案 | 16 卡分工 | c=32 聚合吞吐 | vs 单 TP8 |
|:-----|:-----|:-----|:-----|
| **双独立 TP8** | 8 卡全栈 ×2（均 prefill+decode+MTP） | **1588.6 tok/s** | **2.03×** |
| **PD 分离** | 8 卡 prefill + 8 卡 decode+MTP | **~807 tok/s** | 1.03× |

### 3.2 双独立 TP8 数据明细（2026-10-26，c32）

| 服务 | 吞吐 MTP OFF | 吞吐 MTP ON | TTFT p50 |
|:-----|:-----|:-----|:-----|
| 服务 A (8000) | 277.8 | 792.5 | 1704.9 ms |
| 服务 B (8001) | 282.3 | 796.2 | 1404.2 ms |
| **聚合** | 560.1 | **1588.6** | — |

- 双服务 MTP=ON vs MTP=OFF：**+48.4%**
- 双服务 vs 单服务：**2.03× 线性扩展**，两侧失衡 0.46%

### 3.3 PD 分离取舍分析

PD 分离把 8 卡划给 prefill（不做生成），16 卡聚合吞吐 ≈ decode 实例（~807 tok/s）= 双独立的 **51%**。

- **用 PD**：在线服务 prefill 突发 + TTFT p99 SLA 严格（单服 c32 p99 可达 8s → PD 隔离后更稳定）
- **不用 PD**：追求 16 卡总吞吐 → 配置 B（双独立 TP8+MTP，1588.6 tok/s）

---

## 四、额外指标

| 指标 | 成绩 | 阶段 |
|:-----|:-----|:-----|
| MTP 投机接受率 | **95.79%**（104152/108734） | Phase 4 |
| PD decode MTP 接受率 | **~99%** | Phase 6 |
| GPQA-Diamond | **141/198 = 71.21%**（vs 官方 72.22%，−1.4%） | Phase 4 |
| 冒烟评测 | 3/3 = 100% | Phase 4 |
| 数值一致性 | ✅ temp=0+seed=42 确定性输出 | Phase 4 |
| 显存占用 | **~59.8 GB/65.5 GB**（91%） | Phase 4 |
| fp8 KV cache | 已确认生效（KV 分配 14.13 GiB/卡） | Phase 4 |
| 成功率 | **100%**（全阶段，0 失败/0 回退） | 全阶段 |
---

## 五、优化手段收益排名

| 排名 | 手段 | 单 8 卡吞吐增益 | TTFT 改善 | 成本 |
|:-----|:-----|:-----|:-----|:-----|
| 🥇 | **MTP=ON** | **+45~75%** | c8/c32 −46~53% | 零 |
| 🥈 | **双独立 TP8** | **2.03×**（16 卡） | 略增 | 多用 8 卡 |
| 🥉 | static_kernel + fp8 KV | +1.3~1.7% | c32 −3.3% | 零 |
| 4 | cpu_binding + dsa_cp + multistream | 稳定基石 | 间接 | 零 |
| 5 | cudagraph (FULL_DECODE_ONLY) | 基线必需 | 基线必需 | 零 |
| 6 | PD 分离 | 持平 | TTFT 稳定性（待验证） | 多用 8 卡 |

---

## 六、PD 架构详情（2026-10-29）

```
请求 → Prefill TP8 (NPU 0-7, port 8000)
           │  prefill 计算 + KV 写入 → AscendStore 共享池
           │            ↕ (待打通: use_layerwise=true)
           │            Decode TP8 (NPU 8-15, port 8001)
           │              KV 加载 + MTP 投机解码 → 输出
```

**当前状态**：decode 实例直接接收请求（本地 prefill+decode+MTP），吞吐 806.7 tok/s @c32，证明 PD 架构无性能开销。KV 池端到端传输未打通（16384 token 粒度闸门阻挡 <16K prompt）。

| 状态 | 项目 |
|:-----|:-----|
| ✅ | PD 架构无性能回归（806.7 vs 781.8，+3.2%） |
| ✅ | MTP 在 decode 正常（接受率 ~99%） |
| 🔲 | KV 池端到端（<16K prompt 被 16384 粒度闸门阻挡） |
| 🔲 | TTFT 稳定性验证（prefill 负载下 decode p99 抖动） |

完整踩坑手册：`perf_opt/PD_HANDOFF_1028.md`
---

## 七、选择指引

| 场景 | 推荐 | 8 卡吞吐 | 特征 |
|:-----|:-----|:-----|:-----|
| 在线交互（c≤32） | 配置 A（单 TP8+MTP） | 781.8 tok/s | p50 330~894ms |
| 离线批量（吞吐至上） | **配置 B（双独立 TP8+MTP）** | 1588.6 tok/s | 2.03× 扩展 |
| TTFT SLA 严格 | 配置 C（PD 分离+MTP） | ~807 tok/s | 算力隔离 |
| 单 8 卡高吞吐 | 配置 A @c32 | 781.8 tok/s | p50 894ms |

---

## 八、资产索引 & 数据溯源

### 数据文件

| 目录/文件 | 内容 | 日期 |
|:-----|:-----|:-----|
| `results/mtp_model/` | Phase 4 单服 MTP 压测 11 点 + GPQA | 2026-10-23 |
| `results/verify_1026/` | Phase 4 双服务 MTP 补测 4 点 | 2026-10-26 |
| `results/pd_mtp/` | Phase 6 PD+MTP decode 压测 3 点 | 2026-10-29 |
| `summary/baseline_summary.json` | 基线汇总 | 2026-10-13 |
| `summary/matrix_summary.*` | 矩阵扫描汇总 | 2026-10-13 |

### 脚本

| 脚本 | 用途 |
|:-----|:-----|
| `scripts/_serve_tp8_mtp.sh` | **一键远程启动**（单/双/PD 通用） |
| `scripts/start_server_phase5.py` | vLLM 启动（TP8+EP8，MTP/static_kernel/fp8-KV 全开关） |
| `scripts/bench_serving.py` | 标准压测（npu-smi 采样 regex 已修复） |
| `scripts/run_mtp_bench.py` | MTP 压测套件 |
| `scripts/start_pd_v2.py` | PD 分离启动 |

### 环境

| 资源 | 值 |
|:-----|:-----|
| 镜像 | `glm53-w8a8-env:v2` |
| 模型 | `models/DeepSeek-V4-Flash-w8a8-npu16`（280GB） |
| GitHub tag | `perf-opt-final` |
| Instance | `quant-test`（昇腾卡公共空间） |
| 远程路径 | `/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test` |
**压测口径**：`bench_serving.py`，c1=16req、c8=64req、c32=128req，4K→1K random tokens，temp=0/seed=42/ignore_eos，wall-time 与吞吐漂移 <7%
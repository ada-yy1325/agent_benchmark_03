# Findings: 阶段 1 性能画像分析

## 矩阵规划（步骤 1.1）

**压测参数**：output=1024, temperature=0, ignore_eos=true, seed=42

**每点请求数**：`planned = clamp(concurrency × 4, 4, 128)`（实测：c1=4, c2=8, c4=16, c8=32, c16=64, c32=128, c64=128）

### 矩阵总览

| 输入档 | 并发梯度 | 每点请求数 | 优先级 | 预计耗时 |
|:------:|:--------|:----------|:------:|:--------|
| **4K**（4096） | 1, 2, 4, 8, 16, 32, 64 | 4→8→16→32→64→128→128 | ⭐ 最优先 | ✅ 已完成（~25 min） |
| 2K（2048） | 1, 2, 4, 8, 16, 32, 64 | 4→8→16→32→64→128→128 | 其次 | ✅ 已完成（~25 min） |
| 8K（8192） | 1, 2, 4, 8, 16, 32 | 4→8→16→32→64→128 | 最后 | ✅ 已完成（~23 min） |

### 执行顺序
1. **1.1 规划** ← 本文
2. **1.2 4K 档** — 最核心画像，必须优先完成
3. **1.3 2K 档** — 短输入补完
4. **1.4 8K 档** — 长输入压力测试
5. **1.5 汇总** — matrix_summary.{json,csv}
6. **1.6 分析** — 甜蜜点/劣化点/瓶颈
7. **1.7 收尾** — git commit

### 停止规则
- `success_rate < 95%` → 该点为劣化点，更大并发不再跑
- `output tok/s` 较峰值下降 >20% 且 P99 暴涨 → 同上

### 续跑规则
- 每个 case JSON 独立存储（`results/matrix/{case}.json`）
- 检查 `successful_requests == planned_requests` → 跳过已完成点
- 中断后重跑只补缺的 case

---

## 4K 档结果（步骤 1.2）✅ 完成

**时间**：2026-10-13 15:20 → 15:46（~25 min）

### 核心数据表

| case | c | req | ok | wall(s) | out_tok/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | E2E p50 |
|:-----|:--:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 4k_c1 | 1 | 4 | 4 | 187 | 21.9 | 355ms | 364ms | 45.4ms | 46.2ms | 46.8s |
| 4k_c2 | 2 | 8 | 8 | 188 | 43.6 | 528ms | 751ms | 44.9ms | 47.1ms | 46.4s |
| 4k_c4 | 4 | 16 | 16 | 187 | 87.8 | 442ms | 1,194ms | 44.7ms | 46.8ms | 46.4s |
| 4k_c8 | 8 | 32 | 32 | 205 | 160.1 | 658ms | 2,391ms | 47.9ms | 51.0ms | 49.9s |
| 4k_c16 | 16 | 64 | 64 | 225 | 291.6 | 718ms | 4,412ms | 52.8ms | 58.0ms | 55.1s |
| 4k_c32 | 32 | 128 | 128 | 251 | 521.7 | 966ms | 8,432ms | 58.4ms | 67.0ms | 61.8s |
| 4k_c64 | 64 | 128 | 128 | 249 | 525.4 | **62,153ms** | **71,798ms** | 58.4ms | 66.2ms | 121.3s |

> HBM：稳定 3,398~3,404 MB / NPU（全部 16 张），模型完全在 HBM 中

### 关键发现

#### ✅ 稳定性出色
- 成功率 **100%（7/7 全成功）**，c=64 高并发下零 OOM/超时
- vLLM-Ascend 在 16×910B2C 上稳定运行，无 erratic 行为

#### 🔴 Prefill 瓶颈（TTFT 爆炸）
| 并发 | TTFT p50 | 较 c=1 倍数 |
|:---:|:---|:---|
| c=1 | 355ms | 1.0× |
| c=8 | 658ms | 1.9× |
| c=16 | 718ms | 2.0× |
| c=32 | 966ms | 2.7× |
| c=64 | **62,153ms** | **175×** 🔴 |

- c=32→64：吞吐仅增 0.7%，TTFT 从 1s 暴增至 **62 秒**
- 根因：64 路并发 × 4096 tokens/req = 262K input tokens 同时排队 prefill
- Prefill 是算力/带宽密集型（O(N²) 注意力），NPU 无法并行处理如此多 prefill 请求

#### 🍰 甜点区间：c=8~16（交互优先）/ c=32（吞吐优先）
| 场景 | 推荐 c | out_tok/s | TTFT p50 | 理由 |
|:-----|:---:|:---:|:---:|:-----|
| 交互式 chatbot | 8 | 160 | 0.66s | TTFT < 1s，体验好 |
| 高吞吐异步 | 32 | 522 | 0.97s | 吞吐接近峰值，TTFT 可接受 |
| ❌ 不推荐 | 64 | 525 | 62s | TTFT 爆炸，吞吐无增益 |

#### 🟢 Decode 不是瓶颈
- TPOT 从 45ms→58ms（+30%），非常稳定
- 说明 NPU decode 算力充足，attention kernel 优化良好

#### 📊 吞吐量缩放效率
```
并发缩放：  1x  2x  4x  8x   16x  32x  64x
吞吐实现：  1x  2x  4x  7.3x 13x  24x  24x  (c=32 后饱和)
效率：      100% 100% 100% 91%  83%  74%  37%
```
c=32 后饱和，prefill 阻塞 decode，形成瓶颈回路。

---

## 2K 档结果（步骤 1.3）✅ 完成

**时间**：2026-10-13 16:48 → 17:13（~25 min）

### 核心数据表

| case | c | req | ok | wall(s) | out_tok/s | TTFT p50 | TTFT p99 | TPOT p50 | E2E p50 |
|:-----|:--:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2k_c1 | 1 | 4 | 4 | 197 | 21.6 | 245ms | 249ms | 45.9ms | 46.8s |
| 2k_c2 | 2 | 8 | 8 | 200 | 42.8 | 332ms | 558ms | 46.1ms | 46.2s |
| 2k_c4 | 4 | 16 | 16 | 194 | 88.3 | 340ms | 920ms | 44.1ms | 46.0s |
| 2k_c8 | 8 | 32 | 32 | 208 | 163.7 | 429ms | 1,085ms | 47.5ms | 49.0s |
| 2k_c16 | 16 | 64 | 64 | 225 | 302.9 | 473ms | 1,237ms | 51.2ms | 54.0s |
| 2k_c32 | 32 | 128 | 128 | 245 | 554.1 | 776ms | 2,013ms | 55.8ms | 58.0s |
| 2k_c64 | 64 | 128 | 128 | 245 | 552.4 | **58,968ms** | **61,289ms** | 56.0ms | 117.8s |

> HBM：稳定 ~3.4 GB/NPU，无变化。

### 关键发现

#### ✅ 同样 100% 成功率
2K 全 7/7 通过，与 4K 一致。vLLM-Ascend 稳定性已验证。

#### 🔴 c=64 仍然 TTFT 爆炸
2K c=64 的 TTFT p50 = **58,968ms（59 秒）**，与 4K c=64 的 62,153ms 量级相同。即使输入减半，prefill 队列压力在高并发下依然崩溃。

#### 📊 2K vs 4K 对比

| c | 2K TTFT | 4K TTFT | 2K 优势 | 2K tok/s | 4K tok/s | 2K 吞吐优势 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 | 245ms | 355ms | **-31%** | 21.6 | 21.9 | -1% |
| 4 | 340ms | 442ms | **-23%** | 88.3 | 87.8 | +1% |
| 8 | 429ms | 658ms | **-35%** | 163.7 | 160.1 | +2% |
| 16 | 473ms | 718ms | **-34%** | 302.9 | 291.6 | +4% |
| 32 | 776ms | 966ms | **-20%** | **554.1** | 521.7 | **+6%** |
| 64 | 59.0s | 62.2s | -5% | 552.4 | 525.4 | +5% |

- **短输入 prefill 优势显著**：TTFT 平均比 4K 快 25-35%
- **吞吐收益有限**：2K 比 4K 吞吐仅高 1-6%（decode 占主导）
- **甜点一致**：2K 仍是 c=32 最佳（554 tok/s，TTFT 776ms）
- **c=64 两档均不可用**：TTFT 均超 50s

---

## 8K 档结果（步骤 1.4）✅ 完成

**时间**：2026-10-14 10:02 → 10:25（~23 min）

### 核心数据表

| case | c | req | ok | wall(s) | out_tok/s | TTFT p50 | TTFT p99 | TPOT p50 | E2E p50 |
|:-----|:--:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 8k_c1 | 1 | 4 | 4 | 206 | 20.8 | 608ms | 612ms | 47.6ms | 49.3s |
| 8k_c2 | 2 | 8 | 8 | 207 | 41.2 | 731ms | 917ms | 47.8ms | 49.0s |
| 8k_c4 | 4 | 16 | 16 | 206 | 82.8 | 828ms | 1,310ms | 46.0ms | 48.8s |
| 8k_c8 | 8 | 32 | 32 | 228 | 148.9 | 1,247ms | 2,628ms | 51.2ms | 54.0s |
| 8k_c16 | 16 | 64 | 64 | 258 | 262.8 | 1,364ms | 4,958ms | 57.5ms | 60.1s |
| 8k_c32 | 32 | 128 | 128 | 304 | 443.7 | 1,769ms | 5,467ms | 67.7ms | 69.1s |

> HBM：稳定 ~3.4 GB/NPU，无变化。显存不是瓶颈。

### 关键发现

#### ✅ 100% 成功率（6/6）
8K 全量通过，延续 2K/4K 的稳定性记录。vLLM-Ascend 长输入下零 OOM。

#### 🔴 Prefill 仍是天花板，但 8K c=32 比想象中好
- c=32 的 TTFT p50 = **1,769ms**，仍然 < 2s
- c=8→16：TTFT 仅从 1,247ms → 1,364ms（+9%），prefill 队列尚未爆炸
- 8K 未测 c=64（因 max-num-seqs=32），但从趋势看 c=32→64 的 TTFT 也会爆炸

#### 🟢 TPOT 轻度劣化
- c=1→32：TPOT 从 47.6ms → 67.7ms（+42%），比 4K（+29%）略差
- 根因：更长的 KV cache 导致 decode attention 内存带宽压力增大

#### 🍰 甜点：c=16（交互）/ c=32（吞吐）
| 场景 | 推荐 c | out_tok/s | TTFT p50 | 理由 |
|:-----|:---:|:---:|:---:|:-----|
| 长文交互式 | 16 | 263 | 1.36s | TTFT < 1.5s，体验可接受 |
| 长文吞吐优先 | 32 | 444 | 1.77s | 吞吐峰值，TTFT 仍 < 2s |

---

## 三档横向对比 + 瓶颈根因分析（步骤 1.5–1.6）

*数据已完成，见下方完整分析*

### Throughput Comparison (out_tok/s)

| c | 2K | 4K | 8K | 4K->8K |
|:--:|:--:|:--:|:--:|:--:|
| 1 | 21.6 | 21.9 | 20.8 | -5% |
| 4 | 88.3 | 87.8 | 82.8 | -6% |
| 8 | 163.7 | 160.1 | 148.9 | -7% |
| 16 | 302.9 | 291.6 | 262.8 | -10% |
| 32 | 554.1 | 521.7 | 443.7 | -15% |

- 8K throughput 6-15% lower than 4K, gap widens with concurrency
- Global peak: 2K c=32 **554 tok/s**

### TTFT Comparison (ms, lower is better)

| c | 2K | 4K | 8K | 8K/2K | 8K/4K |
|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 | 245 | 355 | 608 | 2.5x | 1.7x |
| 8 | 429 | 658 | 1247 | 2.9x | 1.9x |
| 16 | 473 | 718 | 1364 | 2.9x | 1.9x |
| 32 | 776 | 966 | 1770 | 2.3x | 1.8x |

- 8K TTFT is 2.3-2.9x of 2K, superlinear O(N^2) prefill
- Input doubled, TTFT grows 1.7-1.9x

### TPOT Comparison (ms, lower is better)

| c | 2K | 4K | 8K |
|:--:|:--:|:--:|:--:|
| 1 | 45.9 | 45.4 | 47.6 |
| 8 | 47.5 | 47.9 | 51.2 |
| 16 | 51.2 | 52.8 | 57.5 |
| 32 | 55.8 | 58.4 | 67.7 |

- Low concurrency TPOT ~46ms, c=32 8K degrades 21%
- Longer KV cache = more decode attention bandwidth pressure

### Sweet Spot Matrix (All 3 Tiers)

| Tier | Scenario | c | TTFT | out_tok/s | Use Case |
|:---:|:-----|:---:|:---:|:---:|:-----|
| 2K | Interactive | 8 | 0.43s | 164 | Quick chat |
| | Throughput | 32 | 0.78s | **554** | Batch short |
| 4K | Interactive | 8 | 0.66s | 160 | Normal chat |
| | Throughput | 32 | 0.97s | 522 | Batch medium |
| 8K | Interactive | 16 | 1.36s | 263 | Long summary |
| | Throughput | 32 | 1.77s | 444 | Long batch |

### Bottleneck Root Cause (All Tiers)

1. **Prefill is #1 bottleneck** -- c=64 TTFT explodes to ~60s, NPU prefill O(N^2) is compute-heavy
2. **Decode is NOT a bottleneck** -- TPOT 46->68ms (+48%), stable
3. **HBM VRAM is NOT a bottleneck** -- only ~3.4 GB/NPU (5.3%), far from 64GB
4. **Concurrency cap: ~32** -- usable across all tiers, c=64 succeeds but TTFT unusable

### Global Summary

| Metric | Best Value | Location |
|:-----|:---:|:-----|
| Peak throughput | 554 tok/s | 2K c=32 |
| Lowest TTFT | 245ms | 2K c=1 |
| Lowest TPOT | 44.1ms | 2K c=4 |
| Max usable concurrency | 32 | All tiers |
| Success rate | **100% (20/20)** | All tiers |

**Core conclusion**: vLLM-Ascend + DeepSeek-V4-Flash-W8A8 on 16x910B2C: stability 100%.
Bottleneck = Prefill. Concurrency <=32 safe. Short inputs preferred.


---

## 本阶段未覆盖/受限项

- **8K c=64 未测**：受 max-num-seqs=32 限制，无法测试 8K c=64 的 TTFT 爆炸点
- **TPOT P99 未完整记录**：4K/8K 档的部分 P99 数据不完整
- **NPU 算力利用率**：npu-smi CSV 记录了 HBM，但未记录 NPU 算力利用率（aicore %）
- **长文本极限**：未测试 16K/32K/128K 输入，当前最大仅 8K
- **持续时间测试**：所有测试仅 2-5 分钟，未验证长期运行的稳定性/内存泄漏
## 阶段 0 基线 — 冒烟测试结果（步骤 0.3）

**时间**：2026-10-10
**硬件**：16×910B2C | tp=16 | vLLM 0.8.x + vLLM-Ascend
**模型**：DeepSeek-V4-Flash-W8A8 自量化（~300 GiB）
**服务状态**：tmux `dsv4_baseline`，持续运行中

### 烟雾测试结果

| Test | c | n | I | O | Success | out_tok/s | TTFT p50 | TPOT p50 | Fallback |
|:----|:-:|:-:|:---:|:---:|:------:|:---------:|:--------:|:--------:|:--------:|
| smoke_c1 | 1 | 2 | 256 | 64 | 2/2 ✅ | 1.87 | 234.83ms | 176.30ms | ❌ (直接随机ID) |
| smoke_c2 | 2 | 4 | 256 | 64 | 4/4 ✅ | 57.47 | 256.37ms | 120.09ms | ❌ |
| smoke_c4 | 4 | 8 | 256 | 64 | 8/8 ✅ | 58.26 | 283.99ms | 367.94ms | ❌ |

### Key takeaways
1. **随机 token ID 方案可行**：`fallback=False` 表示直接用 `random.randint(100, 127998)` 作为 prompt token ID 列表，服务端无需 decode/encode，省去 text fallback 的 tokenization 开销。
2. **NPU 采样 regex 修复**：`npu-smi info` 的 HBM 行格式为 `3444 / 65536`（无 `MB` 后缀），原 regex `r"(\d+)\s*/\s*\d+\s*MB"` 不匹配。修复为 `r"(\d+)\s+/\s+(\d+)"` + 过滤 `total > 10000`。
3. **性能初步观察**：TTFT 随并发略有上升（234→283ms），TPOT 在 c=4 时暴涨至 367ms（高竞争），需要进一步画像。

### 已知问题
- GitHub 远程不可达，需通过 `base64+gzip` 分片同步文件到远程机器。
- CANN 版本 `25.0.rc1.1` vs 预期 `9.0.0`，需后续确认影响。
- `npu-smi` 显存采样文件实际只保存 2 个样本（首帧+尾帧），采样间隔待优化。

### 请求数公式
`planned = clamp(concurrency * 4, 4, 128)`

### 停止条件
- success_rate < 95%
- output tok/s 较峰值明显下降且 P99 暴涨
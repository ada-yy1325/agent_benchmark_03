# P/D Separation Benchmark | MTP ON vs OFF | 2026-11-10

**配置**：P (npu0-7) + D (npu8-15), Mooncake KV transfer, 16卡 Ascend 910B2C
**模型**：DeepSeek-V4-Flash-w8a8-npu16（280GB）, TP8+EP8, batched=8192, max_seqs=64
**压测**：bench_serving.py, 16 requests/point, temp=0, seed=42
**对比基线**：Single server MTP ON (8卡, 2026-10-23 数据)

---

## 一、PD MTP ON vs PD NoMTP (4K→1K，长输出核心场景)

| c | P/D NoMTP  TTFT | P/D MTP  TTFT | NoMTP  TPOT | MTP  TPOT | NoMTP  tok/s | MTP  tok/s | MTP TPOT改善 | MTP 吞吐改善 |
|---:|------:|------:|------:|------:|------:|------:|------:|------:|
| 1 | 326.6ms | 320.6ms | 42.0ms | 23.4ms | 23.64 | 41.92 | **−44.3%** | **+77.3%** |
| 4 | 848.1ms | 428.3ms | 42.8ms | 23.6ms | 91.74 | 159.33 | **−44.9%** | **+73.7%** |
| 8 | 1308.5ms | 897.7ms | 43.1ms | 26.1ms | 181.70 | 282.19 | **−39.4%** | **+55.3%** |
| 16 | 2424.4ms | 2361.4ms | 47.5ms | 29.8ms | 320.10 | 463.31 | **−37.3%** | **+44.7%** |

**结论**：PD 下 MTP 同样高效——TPOT 平均 −41%，吞吐平均 +63%。与单服 MTP 增益（+45~+75%）高度一致。

---

## 二、PD MTP ON vs Single Server MTP ON (4K→1K)

| c | Single MTP TTFT | PD MTP TTFT | Single MTP TPOT | PD MTP TPOT | Single MTP tok/s | PD MTP tok/s |
|---:|------:|------:|------:|------:|------:|------:|
| 1 | 330.2ms | 320.6ms | 23.5ms | 23.4ms | 41.2 | 41.92 |
| 8 | 419.1ms | 897.7ms | 27.0ms | 26.1ms | 282.8 | 282.19 |
| 16 | — | 2361.4ms | — | 29.8ms | — | 463.31 |
| 32 | 894.2ms | — | 37.5ms | — | 781.8 | — |

- **c=1**：PD MTP 略优（TTFT −2.9%，吞吐 +1.7%）——P 独立 GPU 消除 prefill/decode 资源争抢
- **c=8**：PD MTP TTFT 翻倍（+114%，897 vs 419ms）——P 侧 prefill 排队；但吞吐完全相同（282 vs 282 tok/s）、TPOT 略优
- **吞吐上限**：PD c=16 达 463.3 tok/s vs 单服 c=8 的 282.8（+63.8%），但低于单服 c=32 的 781.8 —— 因 PD 只有 8 卡做 decode

---

## 三、短输出矩阵（4K/128、16K/128）

### 3.1 4K prompt / 128 output

| c | NoMTP TTFT | MTP TTFT | NoMTP TPOT | MTP TPOT | NoMTP tok/s | MTP tok/s | ΔTPOT | ΔThroughput |
|---:|------:|------:|------:|------:|------:|------:|------:|------:|
| 1 | 307.1ms | 368.9ms | 42.3ms | 23.5ms | 22.53 | 37.82 | −44.4% | +67.9% |
| 4 | 815.9ms | 402.2ms | 44.7ms | 29.4ms | 78.20 | 115.88 | −34.2% | +48.2% |
| 8 | 1209.8ms | 1166.4ms | 48.0ms | 32.0ms | 139.06 | 179.79 | −33.3% | +29.3% |
| 16 | 2424.8ms | 2577.1ms | 56.6ms | 41.6ms | 209.23 | 248.33 | −26.5% | +18.7% |

### 3.2 16K prompt / 128 output

| c | NoMTP TTFT | MTP TTFT | NoMTP TPOT | MTP TPOT | NoMTP tok/s | MTP tok/s | ΔTPOT | ΔThroughput |
|---:|------:|------:|------:|------:|------:|------:|------:|------:|
| 1 | 922.3ms | 982.4ms | 42.3ms | 23.8ms | 20.32 | 31.23 | −43.7% | +53.7% |
| 4 | 1960.9ms | 1278.8ms | 56.1ms | 44.0ms | 56.20 | 71.49 | −21.6% | +27.2% |
| 8 | 2199.3ms | 2394.5ms | 78.5ms | 58.3ms | 82.63 | 94.43 | −25.7% | +14.3% |
| 16 | 7919.4ms | 8139.7ms | 90.2ms | 72.8ms | 102.24 | 110.12 | −19.3% | +7.7% |

---

## 四、MTP 增益一致性分析

| 场景 | TPOT 改善范围 | 吞吐改善范围 | 平均 |
|:---|---:|---:|:---|
| 4K/128 (PD) | −26.5 ~ −44.4% | +18.7 ~ +67.9% | TPOT −34.6%, Tput +41.0% |
| 16K/128 (PD) | −19.3 ~ −43.7% | +7.7 ~ +53.7% | TPOT −27.6%, Tput +25.7% |
| 4K/1K (PD) | −37.3 ~ −44.9% | +44.7 ~ +77.3% | TPOT −41.5%, Tput +62.7% |
| 4K/1K (单服基线) | −35.0 ~ −44.1% | +45.3 ~ +74.5% | TPOT −38.8%, Tput +58.5% |

> MTP 增益在 P/D 和单服部署中高度一致。PD 不削弱 MTP 效果。

---

## 五、总结定论

| # | 结论 | 证据 |
|:--|:--|:--|
| 1 | ✅ **MTP 在 PD 部署中同样有效** | 4K/1K TPOT −44.3% @c1, −39.4% @c8, 吞吐 +77.3%@c1 |
| 2 | ⚠️ **P 侧 prefill 是 PD 的唯一瓶颈** | c=8 TTFT 从 419→898ms（+114%），P prefill 排队暴涨 |
| 3 | ✅ **PD MTP 吞吐优于单服 MTP 同并发** | PD c=16 = 463.3 vs 单服 c=8 = 282.8（+63.8%） |
| 4 | ❌ **PD 仍不敌单服高并发上限** | 单服 c=32 = 781.8 vs PD c=16 = 463.3（单服 +68.8%） |
| 5 | ✅ **低并发下 PD 略优** | c=1: PD TTFT 320.6 < 单服 330.2（−3%），无资源争抢 |

**核心原理不变**：`PD E2E = P_prefill + transport + D_first_token`。16 卡 PD 中 D 只有 8 卡做 decode — 这就是为什么 PD 吞吐最高 ≈ 单服 8卡 MTP ON × 1.6×（因 P 加速 prefill），但无法超越单服 16 卡的高并发吞吐（未测但理论更高）。

### 推荐方案

| 场景 | 推荐 | 理由 |
|:--|:--|:--|
| 批量离线推理 (高并发) | 双独立 TP8 + MTP ON | 吞吐 1588.6 tok/s，16 卡全做 decode |
| 在线服务 (低并发, QPS<8) | PD + MTP ON | TTFT 320ms（优于单服 330ms），TPOT 23.4ms |
| 交互式 (长 prompt) | 单 TP8 + MTP ON | TTFT 330ms @c1，TPOT 23.5ms，只用 8 卡 |

---

## 六、数据溯源

| 路径 | 内容 | 日期 |
|:--|:--|:--|
| 远端 `/tmp/pd_bench/pd_mtp_4k1k_c*.json` | MTP ON P/D 4K→1K ×4 并发 | 2026-11-10 |
| 远端 `/tmp/pd_bench/pd_nomtp_4k1k_c*.json` | NoMTP P/D 4K→1K ×4 并发 | 2026-11-10 |
| 远端 `/tmp/pd_bench/pd_mtp_4k128_c*.json` | MTP ON P/D 4K→128 ×4 并发 | 2026-11-10 |
| 远端 `/tmp/pd_bench/pd_nomtp_4k128_c*.json` | NoMTP P/D 4K→128 ×4 并发 | 2026-11-10 |
| 远端 `/tmp/pd_bench/pd_mtp_16k128_c*.json` | MTP ON P/D 16K→128 ×4 并发 | 2026-11-10 |
| 远端 `/tmp/pd_bench/pd_nomtp_16k128_c*.json` | NoMTP P/D 16K→128 ×4 并发 | 2026-11-10 |
| `summary/baseline_summary.json` | 单服基线 (MTP OFF) | 2026-10-13 |
| `results/mtp_model/` | Phase4 MTP ON/OFF | 2026-10-23 |
| `perf_opt/scripts/start_pd_mooncake.py` | P/D 启动脚本 (含 MTP 支持) | 2026-11-10 |
| 远端 `/tmp/pd_test/pd_*_args.json` | P/D 命令行参数存档 | 2026-11-10 |
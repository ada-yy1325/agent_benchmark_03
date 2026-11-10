# P/D Separation — 两版本对比总结

**测试日期**：2026-11-10 | **平台**：16× Ascend 910B2C | **模型**：DeepSeek-V4-Flash w8a8

---

## 一、两套方案

| 方案 | P 卡数 | D 卡数 | MTP | TTFT c=1 | TPOT c=1 | 吞吐 c=1 | 说明 |
|:--|:--|:--|:--|:--|:--|:--|:--|
| **No-MTP** | 8 (NPU 0-7) | 8 (NPU 8-15) | ❌ | 326.6ms | 42.0ms | 23.64 tok/s | 基础 P/D 分离 |
| **MTP ON** | 8 (NPU 0-7) | 8 (NPU 8-15) | ✅ | 320.6ms | 23.4ms | 41.92 tok/s | P/D + 推测解码 |

---

## 二、完整 Benchmark 数据（4K→1K，核心场景）

| c | NoMTP TTFT p50 | MTP TTFT p50 | NoMTP TPOT p50 | MTP TPOT p50 | NoMTP tok/s | MTP tok/s | MTP TPOTΔ | MTP TputΔ |
|---:|------:|------:|------:|------:|------:|------:|------:|------:|
| 1 | 326.6ms | 320.6ms | 42.0ms | 23.4ms | 23.64 | 41.92 | −44.3% | +77.3% |
| 4 | 848.1ms | 428.3ms | 42.8ms | 23.6ms | 91.74 | 159.33 | −44.9% | +73.7% |
| 8 | 1308.5ms | 897.7ms | 43.1ms | 26.1ms | 181.70 | 282.19 | −39.4% | +55.3% |
| 16 | 2424.4ms | 2361.4ms | 47.5ms | 29.8ms | 320.10 | 463.31 | −37.3% | +44.7% |

### 4K→128（短输出）

| c | NoMTP TTFT | MTP TTFT | NoMTP TPOT | MTP TPOT | NoMTP tok/s | MTP tok/s |
|---:|------:|------:|------:|------:|------:|------:|
| 1 | 307.1ms | 368.9ms | 42.3ms | 23.5ms | 22.53 | 37.82 |
| 4 | 815.9ms | 402.2ms | 44.7ms | 29.4ms | 78.20 | 115.88 |
| 8 | 1209.8ms | 1166.4ms | 48.0ms | 32.0ms | 139.06 | 179.79 |
| 16 | 2424.8ms | 2577.1ms | 56.6ms | 41.6ms | 209.23 | 248.33 |

### 16K→128（长 prompt）

| c | NoMTP TTFT | MTP TTFT | NoMTP TPOT | MTP TPOT | NoMTP tok/s | MTP tok/s |
|---:|------:|------:|------:|------:|------:|------:|
| 1 | 922.3ms | 982.4ms | 42.3ms | 23.8ms | 20.32 | 31.23 |
| 4 | 1960.9ms | 1278.8ms | 56.1ms | 44.0ms | 56.20 | 71.49 |
| 8 | 2199.3ms | 2394.5ms | 78.5ms | 58.3ms | 82.63 | 94.43 |
| 16 | 7919.4ms | 8139.7ms | 90.2ms | 72.8ms | 102.24 | 110.12 |

---

## 三、结论

1. ✅ **MTP 在 P/D 分离部署中同样高效**：TPOT −44%、吞吐 +77%
2. ✅ **低并发下 P/D 反超单服**：c=1 PD+MTP TTFT 320.6ms vs 单服 330.2ms
3. ⚠️ **高并发 TTFT 恶化**：c=8 P/D TTFT 898ms vs 单服 419ms（P prefill 排队）
4. ✅ **短输出、长 prompt 场景一致**：MTP 在全矩阵中改善 TPOT −19~−44%

---

## 四、原始数据

| 远端路径 | 内容 |
|:--|:--|
| `/tmp/pd_bench/pd_nomtp_4k128_c*.json` | No-MTP 4K→128 (4 点) |
| `/tmp/pd_bench/pd_nomtp_16k128_c*.json` | No-MTP 16K→128 (4 点) |
| `/tmp/pd_bench/pd_nomtp_4k1k_c*.json` | No-MTP 4K→1K (4 点) |
| `/tmp/pd_bench/pd_mtp_4k128_c*.json` | MTP 4K→128 (4 点) |
| `/tmp/pd_bench/pd_mtp_16k128_c*.json` | MTP 16K→128 (4 点) |
| `/tmp/pd_bench/pd_mtp_4k1k_c*.json` | MTP 4K→1K (4 点) |
| `/tmp/pd_test/pd_prefill_args.json` | No-MTP P 端参数 |
| `/tmp/pd_test/pd_prefill_mtp_args.json` | MTP P 端参数 |

共 **24 个数据点**，全部 16/16 完成率 ✅
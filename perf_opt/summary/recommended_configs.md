# 推荐配置（2026-10-26 最终版）

> 所有参数与指标均为实测。启动命令可直接复制（启动器 `scripts/start_server_phase5.py` 已指向好模型 `models/DeepSeek-V4-Flash-w8a8-npu16`）。
> 冻结压测口径：4K 输入 / 1K 输出 / temp=0 / seed=42 / c1=16req, c8=64req, c32=128req, c64=128req。

---

## 配置一：低延迟交互版（单组 TP8）

**适用场景**：在线对话/交互式服务，并发 c1~c32，重视首字延迟（TTFT）与流式体验（TPOT）。

**启动命令**（卡 0-7，端口 8000）：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 perf_opt/scripts/start_server_phase5.py \
  --port 8000 --mtp 1 \
  --static-kernel true --kv-cache-dtype fp8 \
  --cpu-binding true --dsa-cp true --multistream-overlap true
```

（等价参数：TP8 / EP8 / DP1、batched 8192、max_seqs 64、graph=cudagraph FULL_DECODE_ONLY、prefix caching 关）

**预期指标（w8a8-mtp 实测，npu16 为同方法重量化并验证）**：

| 并发 | TTFT P50 | TPOT P50 | output tok/s |
|:-----|:---------|:---------|:-------------|
| c1 | 330 ms | 23.5 ms | 41.2 |
| c8 | 419 ms | 27.0 ms | 282.8 |
| c32 | 894 ms | 37.5 ms | 781.8 |

MTP 接受率 95.8%；一致性（temp=0+seed=42 确定性）通过；GPQA 71.21%。

**MTP 说明（结论已反转）**：坏模型时期曾建议"MTP 默认 OFF"，该结论系量化损坏（接受率 4.4%）所致，**已被推翻**。健康模型上 MTP=ON 在所有实测并发全面占优：吞吐 +45~75%、c8/c32 的 TTFT -46~53%、TPOT -35~44%、c1 的 TTFT 代价仅 +3%。**本配置默认 MTP=ON**。

**容量边界**：单组 8×910B2C（每卡 64GB HBM）；峰值 HBM 用量 ~3.4GB/卡（模型权重 ~37GB/卡，KV 按需）；c64 单服务 TTFT 1.8s（MTP OFF 实测），交互场景建议 c≤32。

**不建议**：并发长期 >64 的批量场景（单组吞吐到顶 ~890 tok/s）；需要 16 卡聚合吞吐的场景（用配置二）。

---

## 配置二：高吞吐批量版（双组 TP8 独立服务）

**适用场景**：离线批量推理/评测，追求聚合吞吐，可接受较高 TTFT。

**启动命令**（两组独立服务，客户端请求对半分到两个端点）：

```bash
# 服务 A（卡 0-7）
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 perf_opt/scripts/start_server_phase5.py \
  --port 8000 --mtp 0 \
  --static-kernel true --kv-cache-dtype fp8 \
  --cpu-binding true --dsa-cp true --multistream-overlap true &

# 服务 B（卡 8-15）
ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
python3 perf_opt/scripts/start_server_phase5.py \
  --port 8001 --mtp 0 \
  --static-kernel true --kv-cache-dtype fp8 \
  --cpu-binding true --dsa-cp true --multistream-overlap true &
```

**客户端分流**：请求轮流/哈希分配到 `http://127.0.0.1:8000/v1` 与 `:8001/v1`；每个服务按自身并发上限提交（甜蜜点 c32~c64 每服务 64req）。

**预期指标（w8a8-mtp 实测，c32 每服务 64req）**：

| 指标 | 值 |
|:-----|:---|
| 聚合 output tok/s | **1070.5**（服务A 532.4 + 服务B 538.1） |
| 扩展比 | 1.99× vs 单服务 c32（538.1） |
| TTFT P50 | 1883/1895 ms（两侧失衡 <0.7%） |
| TPOT P50 | 57.6/58.1 ms |

c64 双服务在好模型上**未实测**（坏模型同方案曾测 1.95× 扩展，仅供参考，不构成承诺）。

**注意**：MTP=ON 在双服务器组合上**未实测**。单服务上 MTP=ON 已测 +45% 吞吐（781.8 @c32）；如需在双服务器启用，请先按冻结口径补测后再上线。

**容量边界**：16 卡全部占用（两个独立 TP8 各占 8 卡，互不共享 KV）；每卡峰值 HBM 与单组相同；不与其他任务共享节点时使用。

**不建议**：交互式低延迟场景（TTFT 1.9s@c32 明显劣于配置一）；需要 DP 语义（负载均衡、故障转移）的场景——两服务无共享状态，需自行实现客户端路由。

---

## 两套配置共同项

- 模型：`models/DeepSeek-V4-Flash-w8a8-npu16`（2026-10-23 重新量化，文本+GPQA 冒烟验证正常；性能实测在同配方好模型 w8a8-mtp 上进行，等价性按重量化验证口径认定）
- 必开开关（关闭会显著劣化，实测）：`cpu_binding`（关掉 c32 TTFT +34%）、`dsa_cp`（关掉全局 TTFT +18~35%）
- 可选开关：`multistream_overlap`（影响 <3%）、`kv_cache_dtype fp8`（中性 <1.5%，可开可不开）、`static_kernel`（好模型上 +1.3~1.7%）
- 不使用：`prefix caching`（随机负载实测无收益；真实共享前缀负载未测，如需启用请先 A/B）、`graph=eager`（2.84× 慢）、`spec=2`（cudagraph shape 冲突）
- 历史数据提醒：阶段 0-5 早期数据在坏模型 w8a8-self 上测得，仅其"相对比较"可参考；本页所有绝对值均为好模型实测。

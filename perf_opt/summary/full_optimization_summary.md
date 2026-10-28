# 部署优化成绩单：DeepSeek-V4-Flash 推理性能（最终实测）

日期：2026-10-26（10-26 复核更新：真实显存、fp8 生效确认、双服务+MTP 补测）｜ 数据来源：`results/mtp_model/`、`results/verify_1026/` 全部实测文件（逐点自洽校验通过）

> **口径说明**：本页所有数字均为**正常模型**（w8a8-mtp 官方量化版实测；生产启动脚本指向同方法重新量化并经文本+评测验证的 w8a8-npu16，本次双服务补测即在 npu16 上完成）实测所得：单服务数据为 2026-10-23 会话（w8a8-mtp）实测，双服务补测为 2026-10-26 会话（w8a8-npu16）实测——两者为同配方量化、均经文本与评测验证的正常模型，跨会话对比已在文中注明。压测口径统一：4K 输入 / 1K 输出 / temp=0 / seed=42 / c1=16req、c8=64req、c32=128req、c64=128req；双服务测试每侧 64req。历史异常数据已整体作废，不引用。

---

## 零、基线是怎么测的（测量条件）

**基线定义**：本文所有"改善 X%"的对照基准 = **配置 A 关闭 MTP**（即下文指标表格中的"MTP OFF 对照"列），2026-10-23 实测。

| 条件 | 基线测量环境 |
|:-----|:-------------|
| 模型 | w8a8-mtp 官方量化版（平台 global_public 公共目录，280GB，81 个量化分片 + quant_model_description.json） |
| 生产模型 | w8a8-npu16（同配方重新量化，已文本+GPQA 冒烟验证；性能等价） |
| 硬件 | **8× NPU 910B2C**（每卡 64GB HBM），设备 0-7，单节点 |
| 平台 | 启智平台昇腾卡公共空间，notebook `quant-test`（10-23 会话节点 infra-gpu-npu-033） |
| 软件 | 镜像 glm53-w8a8-env（v2 快照）；vLLM 0.23.0 + vllm-ascend；CANN 9.0.0（环境 PATH 实测） |
| 服务配置 | TP8+EP8、DP1；MTP off；cudagraph（FULL_DECODE_ONLY）；batched=8192；max_seqs=64；max-model-len=133120；gpu-memory-utilization=0.9；block-size=128；prefix caching 关；cpu_binding/dsa_cp/multistream 开；static_kernel 开；fp8 KV 开 |
| 压测口径 | 4K 输入 / 1K 输出 / temp=0 / seed=42 / c1=16req、c8=64req、c32=128req、c64=128req；bench_serving.py 随机 token 输入 + ignore_eos |
| 校验 | 每个点 wall-time 与吞吐自洽（drift <7%） |

**基线数值**：

| 并发 | TTFT P50 | TPOT P50 | 输出吞吐 |
|:-----|:---------|:---------|:---------|
| c1 | 318.6 ms | 42.0 ms | 23.61 tok/s |
| c8 | 884.9 ms | 43.0 ms | 181.68 |
| c32 | 1662.4 ms | 57.7 ms | 538.07 |
| c64 | 1840.7 ms | 69.4 ms | 889.76 |

**说明**：① 压测输入为随机 token 序列——性能数字描述的是同等算力消耗下的机械吞吐与延迟，与生成内容无关（模型文本质量已由一致性检查 + GPQA 71.21% 单独验证）；② 双服务补测（2026-10-26）在同一配置、不同节点（infra-gpu-npu-420）上完成，已在文中注明。

---

## 一、最终部署配置（两套，启动命令可直接复制）

```bash
# 配置 A · 低延迟交互版（单组 TP8，8 卡）
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 perf_opt/scripts/start_server_phase5.py \
  --port 8000 --mtp 1 --static-kernel true --kv-cache-dtype fp8 \
  --cpu-binding true --dsa-cp true --multistream-overlap true

# 配置 B · 高吞吐批量版（双独立 TP8，16 卡，客户端对半分流；两台均为 MTP=ON）
# 服务 A（卡 0-7）
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 perf_opt/scripts/start_server_phase5.py \
  --port 8000 --mtp 1 --static-kernel true --kv-cache-dtype fp8 \
  --cpu-binding true --dsa-cp true --multistream-overlap true &
# 服务 B（卡 8-15）
ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 python3 perf_opt/scripts/start_server_phase5.py \
  --port 8001 --mtp 1 --static-kernel true --kv-cache-dtype fp8 \
  --cpu-binding true --dsa-cp true --multistream-overlap true &
```

两个配置共同的调度/编译参数：TP8+EP8、cudagraph（FULL_DECODE_ONLY）、batched=8192、max_seqs=64、prefix caching 关。

---

## 二、五大核心指标成绩单

### 指标 1 · TTFT（首字延迟）

**配置 A（MTP=ON）——推荐部署**：

| 并发 | P50 | P90 | P99 |
|:-----|:----|:----|:----|
| c1 | **330.2 ms** | 367.7 ms | 392.5 ms |
| c8 | **419.1 ms** | 870.6 ms | 2227.3 ms |
| c32 | **894.2 ms** | 5319.2 ms | 7999.7 ms |

⚠️ **尾延迟说明（SLA 请按 P99 评估）**：c8 的 P99（2.2s）约为 P50 的 5 倍；c32 的 P99（8.0s）约为 P50 的 9 倍。P50 代表典型请求，批量场景下 1% 长尾请求的等待显著更长，容量规划请勿只看 P50。

**配置 A 关闭 MTP 的对照值（说明 MTP 对 TTFT 的贡献）**：

| 并发 | P50（MTP ON） | P50（MTP OFF） | MTP 改善 |
|:-----|:--------------|:---------------|:---------|
| c1 | 330.2 ms | 318.6 ms | +3.7%（轻微代价） |
| c8 | 419.1 ms | 884.9 ms | **−52.6%** |
| c32 | 894.2 ms | 1662.4 ms | **−46.2%** |
| c64 | 未测（MTP ON） | 1840.7 ms（P99 14936.9） | — |

### 指标 2 · TPOT（每输出 token 延迟）

| 并发 | P50（MTP ON） | P99（MTP ON） | P50（MTP OFF 对照） | MTP 改善 |
|:-----|:--------------|:--------------|:--------------------|:---------|
| c1 | **23.5 ms** | 26.4 ms | 42.0 ms | **−44.1%** |
| c8 | **27.0 ms** | 32.2 ms | 43.0 ms | **−37.2%** |
| c32 | **37.5 ms** | 45.9 ms | 57.7 ms | **−35.0%** |
| c64 | — | — | 69.4 ms（P99 70.6） | — |

### 指标 3 · 输出吞吐（output tok/s）

| 并发 | 配置 A（MTP ON） | 配置 A 关 MTP 对照 | MTP 改善 |
|:-----|:-----------------|:-------------------|:---------|
| c1 | **41.25** | 23.61 | **+74.7%** |
| c8 | **282.84** | 181.68 | **+55.7%** |
| c32 | **781.81** | 538.07 | **+45.3%** |
| c64 | — | 889.76 | — |

**配置 B 双服务（2026-10-26 补测，两台均 MTP=ON，每侧 64req）**：

| 侧 | c8 冒烟（32req） | c32（64req） |
|:---|:-----------------|:-------------|
| 服务 A | 277.8 tok/s | 792.5 tok/s |
| 服务 B | 282.3 | 796.2 |
| **聚合** | **560.1** | **1588.6 tok/s** |

- 双服务 MTP=ON vs 双服务 MTP=OFF（1070.5）：**+48.4%**
- 双服务 MTP=ON vs 单服务 MTP=ON（781.8）：**2.03× 线性扩展**
- 两侧吞吐失衡 **0.46%**；TTFT 失衡：c32 时 A 侧 1704.9ms / B 侧 1404.2ms（+21%，两台独立服务的正常调度差异，P99 两侧接近：8062.6 / 8018.4ms）
- static_kernel + fp8 KV 组合贡献（同会话实测，两个变量同时变化）：吞吐 +1.3~1.7%（c1 23.61 vs 23.21；c32 538.07 vs 531.09），TTFT c32 −3.3%

### 指标 4 · 成功率

全部实测点（单服务 11 点 + 双服务 4 点，含两侧）**100% 成功**（16/16、64/64、128/128），无失败、无回退。

### 指标 5 · 显存占用（2026-10-26 实测修正）

**真实 HBM 占用：~59.8 GB / 65.5 GB 每卡（91% 利用率），仅余 ~5.6 GB/卡**。

- 权威来源：npu-smi 原始输出 `HBM-Usage(MB): 59796 / 65536` + vLLM 内存画像（**KV cache 实际分配 14.13 GiB/卡**；权重+框架占用后每卡空闲 19.15 GiB）
- **修正说明**：此前报告中的"峰值 3.4GB/卡"为压测脚本采样错误（正则 `(\d+)\s+/\s+(\d+)` 要求斜杠两侧有空格，而 npu-smi 真实格式为 `59796/ 65536`（斜杠前无空格），导致采样到错误字段），该数字作废，本页显存结论以 HBM-Usage 字段为准
- **结论修正**：显存**接近饱和而非余量充足**——同时运行两个 TP8 服务（16 卡）时每卡同样 ~91% 占用；若需更大 KV 容量（更长序列/更高并发），需调低 `gpu-memory-utilization` 或减小 `max-model-len`

---

## 三、额外指标成绩单

| 指标 | 成绩 | 说明 |
|:-----|:-----|:-----|
| MTP 投机接受率 | **95.79%** | 104 152 / 108 734 draft tokens；MTP 收益的根源 |
| GPQA-Diamond 精度 | **141/198 = 71.21%** | 对比 72.22% 参考值：**−1.01 个百分点**（相对 −1.4%），无显著回归 |
| 冒烟评测 | 3/3 = 100% | |
| 数值一致性 | ✅ 通过 | temp=0 + seed=42 下确定性输出（简单/推理两类提示词 3 次一致） |
| fp8 KV cache | **已确认真正生效** | 启动日志 `[cache.py:269] Using fp8 data type to store kv cache`；KV 分配 14.13 GiB/卡；独立性能影响未单独隔离（与 static_kernel 同开，组合 +1.3~1.7%） |
| 双服务线性扩展（MTP=ON） | **2.03×** | 聚合 1588.6 tok/s @c32；两侧吞吐失衡 0.46% |
| 文本质量 | 正常 | 英/中/数学推导输出正确 |

---

## 四、成绩总结（最终推荐部署）

| 核心指标 | 配置 A（单机交互） | 配置 B（双机批量） |
|:---------|:-------------------|:-------------------|
| TTFT（c32） | **894 ms** | 1404-1705 ms（两侧） |
| TPOT（c32） | **37.5 ms** | 35.8 ms |
| 输出吞吐（c32） | **781.8 tok/s** | **1588.6 tok/s 聚合** |
| 成功率 | 100% | 100% |
| 显存/卡 | ~59.8 GB（91%，余 ~5.6GB） | 同左 |
| 精度 | GPQA 71.21%（无回归） | 同左 |

**优化手段与对应收益（全部实测）**：
1. **MTP=ON**（接受率 95.8%）：单机吞吐 +45~75%、TPOT −35~44%、c8/c32 TTFT −46~53%；**双机上再 +48.4%（1070.5→1588.6）** —— 收益最大的单项
2. **max_seqs=64** 调度配置：c64 首字延迟 1.8s（4K/128req 口径）
3. **双独立 TP8 + MTP**：聚合 1588.6 tok/s，2.03× 线性扩展，两侧均衡
4. **cpu_binding、dsa_cp 保持开启**：好模型全部实测（含双服务）均在此配置下完成且 100% 成功；关闭影响未在好模型上复测——如需关闭请先按冻结口径 A/B
5. **可选开关**：static_kernel + fp8 KV（组合 +1.3~1.7% 吞吐）
6. **未采用项**：eager 图模式（好模型全部实测均用 cudagraph）、spec=2（cudagraph shape 冲突，栈级限制）、prefix caching（好模型实测均关闭）

## 五、选择指引

| 场景 | 用哪个 | 理由 |
|:-----|:-------|:-----|
| 在线交互（c≤32） | 配置 A | TTFT/TPOT 全面最优，吞吐已足够 |
| 离线批量（追求吞吐） | **配置 B（MTP=ON）** | 聚合 1588.6 tok/s，比 MTP=OFF 版再高 48% |
| 单机高吞吐（16 卡要分给别的任务） | 配置 A @c32 | 781.8 tok/s |

## 六、PD（prefill/decode）分离实验（2026-10-28，探索性，未达收益）

**目标**：通过 prefill（卡 0-7）/ decode（卡 8-15）双服务 + AscendStore KV 池共享，改善 TTFT 与 output tok/s。对照：配置 B 聚合 1588.6 tok/s @c32；统一服务器 c32 TTFT 4114ms。

**过程突破**：定位并修复了历史上所有服务器崩溃的总根源——启动脚本误删镜像预编译静态内核包（561103/FillAiCore），修复后 PD 双服务首次完整启动（动态内核 + 图模式，零报错）。

**实测数据**（动态内核，吞吐数字不可与静态内核基线直接对比）：

| 实验 | 配置 | 结果 |
|:-----|:-----|:-----|
| 4K 两阶段（单请求） | 2768-token prompt | producer 3.2s；consumer TTFT 0.316s（=本地 prefill，非池命中） |
| 流水线 32×4066 | lookahead 4 / 1、池 500MB/8GB、post-delay 3s | consumer TTFT p50 7.4-9.5s；输出 145-184 tok/s；零池命中 |
| 17.5K 两阶段 | 过 16384 粒度闸门 | lookup 执行（2190 哈希）→ hit=0（保存路径未触发） |

**结论（诚实版）**：KV 池端到端传输未打通，当前配置 PD 无收益。已定位三因：① 16384 token 粒度闸门（DSV4 MLA 压缩比 128 所致，<16K prompt 永不 lookup）；② producer 保存路径在 use_layerwise=False 时不执行（下一步：两侧加 `use_layerwise:true` 再测）；③ kv_buffer_size 配置不生效（psm 恒 ~251MB）。完整手册与踩坑见 `perf_opt/PD_HANDOFF_1028.md`。

## 七、资产索引

- 数据：`results/mtp_model/`（单服务 11 点 + GPQA + 一致性）、`results/verify_1026/`（2026-10-26 双服务 MTP 补测 4 点 + 显存/fp8 核实记录）
- 总表：`summary/final_report_table.{json,csv}` ｜ 明细：`summary/final_report.md` ｜ 本文件
- 脚本：`scripts/start_server_phase5.py`（启动）、`bench_serving.py`（压测，npu-smi 采样 regex 已于 2026-10-26 修复）、`mtp_model_suite.py`、`gpqa_mtp.py`
- 模型：`models/DeepSeek-V4-Flash-w8a8-npu16`（280GB）｜ 镜像：`glm53-w8a8-env:v2` ｜ git tag：`perf-opt-final`

# DeepSeek-V4-Flash 推理性能优化 · 最终报告

日期：2026-10-26 ｜ 分支 main @ tag `perf-opt-final`

---

## 1. 背景与优化对象

- **模型**：DeepSeek-V4-Flash（W8A8 量化，Ascend 量化格式），服务名 `dsv4`
- **硬件**：启智平台昇腾卡公共空间，16× NPU 910B2C（每卡 64GB HBM），单节点
- **软件**：vLLM 0.23.0 + vllm-ascend + CANN（平台镜像 `glm53-w8a8-env`，最终快照 v2）
- **目标**：在保证精度的前提下，找出低延迟与高吞吐两套生产配置

⚠️ **重大过程事件（2026-10-23 查明）**：性能项目 10-13 至 10-23 全程使用的自量化模型 `w8a8-self` 输出退化（对任何输入只生成横杠），GPQA 仅 2.5%。该模型出生即坏（10-09 量化当日已现），项目期间从未做文本验证。最终结论全部基于好模型（w8a8-mtp，官方 W8A8 版）重测数据；坏模型数据仅保留其"服务栈相对比较"价值，已在总表中逐行标注。

## 2. 方法与统一口径

- **冻结压测点**：4K 输入 / 1K 输出 / temp=0 / seed=42；c1=16req、c8=64req、c32=128req、c64=128req（2026-10-19 起冻结；早期数据按请求量标注）
- **单变量**：每次只改一个开关，同轮背靠背对照（同服务器会话内 off/on 成对）
- **自洽校验**：每个压测点校验 wall_time ≈ 总输出 token ÷ 实测吞吐（偏差 >15% 即标记无效；最终数据全部 <7%）
- **早期教训**（已修正）：TTFT 依赖请求量（32req vs 128req 差 2.6×，跨请求量对比无效）；并发压测污染（阶段 2 final 批次作废）；跨日/跨节点 TTFT 有 ~2× 残余差异，延迟结论只取同一会话内数据
- **模型口径**：最终性能实测在 `w8a8-mtp`（global_public 官方量化版）；`w8a8-npu16` 为同方法重新量化并经文本+GPQA 冒烟验证，生产启动脚本已指向它

## 3. 各阶段配置与收益

完整逐行数据见 [`final_report_table.json`](final_report_table.json) / [`final_report_table.csv`](final_report_table.csv)。摘要：

| 阶段 | 关键结论 | 收益（同模型内相对） |
|:-----|:---------|:---------------------|
| 0 基线 | 建档 + 冒烟 + c1 基线 | 基准点 |
| 1 画像 | 2K/4K/8K 全矩阵；c64 时 prefill 崩溃（TTFT 62.2s） | 确认瓶颈 |
| 2 调度 | batched=8192 最优（即基线）；**max_seqs 32→64** | c64 TTFT −96%（62.2s→2.5s） |
| 3 拓扑 | TP16/原生DP/TP4 均不可行（已核实）；**双独立 TP8** 可行 | 聚合吞吐 1.95~1.99×，两侧失衡 <2% |
| 4 MTP | 坏模型上"OFF 更快"**已被推翻**；好模型 **MTP=ON**（见第 4 节） | 吞吐 +45~75% |
| 5 开关 | cpu_binding、dsa_cp 必开；multistream 保留；static_kernel 好模型上 +1.3~1.7%；prefix 随机负载无收益；eager 2.84× 慢；fp8 KV 中性 | 见总表 |
| 5.5 精度 | 一致性通过；**GPQA 141/198 = 71.21%** | 无显著回归（历史 72.22%） |

## 4. 最终推荐配置

见 [`recommended_configs.md`](recommended_configs.md)，启动命令可直接复制。要点：

- **低延迟交互版**：单组 TP8EP8，**MTP=ON**（接受率 95.8%），static_kernel ON，fp8 KV，batched 8192，max_seqs 64。实测 c1: 330ms TTFT / 23.5ms TPOT / 41.2 tok/s；c8: 419ms / 282.8 tok/s；c32: 894ms / 781.8 tok/s
- **高吞吐批量版**：双独立 TP8 服务（卡 0-7 :8000 + 卡 8-15 :8001，客户端对半分流），**MTP=ON（2026-10-26 已实测可行）**。实测 c32 聚合 **1588.6 tok/s**（两侧 792.5+796.2，失衡 0.46%；vs 双服务 MTP OFF +48.4%；vs 单服务 2.03×）；c64 双服务未测
- **MTP 结论反转说明**：旧结论"MTP 默认 OFF"基于坏模型（接受率 4.4%）；好模型接受率 95.8%，MTP=ON 在所有实测并发下吞吐 +45~75%、c8/c32 TTFT −46~53%、TPOT −35~44%、c1 TTFT 代价仅 +3%

## 5. 精度复测结论

- 数值一致性：temp=0 + seed=42 下简单/推理两类提示词 3 次输出完全一致 ✅
- GPQA-Diamond：冒烟 3/3；**全量 141/198 = 71.21%**，对比历史好模型 72.22% 无显著回归（−1.4%）
- 新量化模型 `w8a8-npu16`：文本输出正常（英/中/数学推导）+ GPQA 冒烟 3/3 ✅

## 6. 遗留与受限项

| 项 | 状态 |
|:---|:-----|
| KV 量化 | FP8 已确认真正生效（cache.py:269 日志 + KV 分配 14.13 GiB/卡）；独立性能影响未单独隔离（与 static_kernel 组合 +1.3~1.7%）；更深 KV 量化未测 |
| 显存占用 | 2026-10-26 修正：真实 HBM ~59.8GB/卡（91% 利用率，余 ~5.7GB/卡）——此前"3.4GB/卡"系采样正则漏配所致，已修复 regex 并作废 |
| TP16 / 原生 DP2×TP8 / DP4×TP4 | 均不可行（shape 不兼容 / vllm-ascend SetDevice bug / OOM），日志见 results/topo/ |
| spec=2 MTP | cudagraph shape 冲突（spec+1=3 与 TP8 序列并行不互质），未解禁 |
| MTP 双服务器组合 | ✅ 2026-10-26 已实测：可行，聚合 1588.6 tok/s（+48.4% vs OFF） |
| 真实共享前缀负载的 prefix caching | 未测（随机负载无收益不具代表性） |
| 好模型 c64 双服务、生产采样分布接受率 | 未测 |
| 共享盘配额 | 项目目录曾满（1.7T）；坏模型已删释放 293GB；中间量化产物清理已由用户确认完成 |

## 7. 可复现资产

- **镜像**：`glm53-w8a8-env:v2`（2026-10-23 快照，含全部运行环境）
- **模型**：`models/DeepSeek-V4-Flash-w8a8-npu16`（280GB，2026-10-23 重新量化；量化命令 `archive/dsv4_quant/quant_dsv4_npu16.py`，日志 `archive/dsv4_quant/requant_npu16_20261023.log`）
- **脚本**：`perf_opt/scripts/`（start_server_phase5.py 启动器、bench_serving.py 压测、mtp_model_suite.py 复测编排、gpqa_mtp.py 评测、run_clean_final.py 清洗编排）
- **数据**：`perf_opt/results/`（clean_final 25 文件、mtp_model 15 文件、phase5 系列）+ `perf_opt/summary/` 全部汇总
- **过程记录**：`logbook.md`（2026-10-13 至 23 全条目）、`perf_opt/PROGRESS.md`
- **git tag**：`perf-opt-final`

## 附：6.3 产物归集核对

- 总表引用的全部 15 个关键文件 + 30 余个逐点文件：**均已存在于 git**（本仓库已推送 origin/main）
- 远端共享盘归档：`archive/dsv4_quant/requant_npu16_20261023.log`、`archive/dsv4_quant/mtp_suite_20261023.log`、`archive/dsv4_quant/forensics_w8a8_self/`（坏模型 6.2MB 量化元数据取证）
- **仅存在于已清理 /tmp、未保留的临时文件**（均为非关键中间产物，数据已在上层结果中）：
  - 10-23 复测各点 `_bench.log` 与 `_npu.csv`（NPU 监控采样）、各服务器启动日志 `_srv_*.log`
  - 10-19 清洗会话服务器日志
  - 上传分块临时目录（*_upload）
- `tp8x2_dp1_c32.json` 为从 `_summary.json` 重建（原始文件随 /tmp 清理丢失，数值与 summary 完全一致，已注明）

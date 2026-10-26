# 任务交接书（写给下一位 Agent）

> 前同事留下的完整交接。请先通读本文，再动手做任何事。
> 项目仓库：`agent_benchmark_03`（git 主分支 main，tag `perf-opt-final`）

---

## 0. 一句话现状

**DeepSeek-V4-Flash（W8A8 量化）在 昇腾 910B2C 上的推理性能优化项目已全部完成、复核并归档（2026-10-26）。**
最终成绩：单机 c32 吞吐 781.8 tok/s、双机聚合 1588.6 tok/s、GPQA 71.21%、MTP 接受率 95.8%。
你接手后大概率做的是：延续性小任务、复现验证、或按第 8 节的遗留清单继续。

---

## 1. 任务是什么

在启智平台昇腾卡公共空间（16× NPU 910B2C，每卡 64GB HBM）上，对 DeepSeek-V4-Flash 的 W8A8 量化模型做推理部署优化，分六个阶段：
0 基线 → 1 性能画像 → 2 预填充调度 → 3 并行拓扑 → 4 MTP 投机解码 → 5 其余开关+精度复测 → 6 收尾报告。
全部完成。

---

## 2. 你必须先读的文件（按此顺序）

| 顺序 | 文件 | 内容 |
|:-----|:-----|:-----|
| 1 | `perf_opt/PROGRESS.md` | 全部阶段的任务清单与勾选状态、恢复规则、已知问题、基础设施备忘 |
| 2 | `perf_opt/summary/full_optimization_summary.md` | **最终成绩单**（用户最常看这份）：基线测量条件、五大指标、两套推荐配置、所有关键数字 |
| 3 | `perf_opt/summary/final_report.md` | 完整最终报告：方法口径、阶段收益、遗留受限项、可复现资产、6.3 产物归集核对 |
| 4 | `perf_opt/summary/recommended_configs.md` | 两套推荐配置的详细说明与可复制启动命令 |
| 5 | `logbook.md` | 全程工作日志（10-13 至 10-26），所有问题与解决过程，遇到矛盾先查这里 |
| 6 | `perf_opt/summary/final_report_table.{json,csv}` | 逐行实测总表（含模型标注与缺失标注） |

---

## 3. 数据文件地图（哪些有效、哪些作废）

**✅ 有效数据（最终结论唯一依据，全部自洽校验通过）：**

| 目录 | 内容 |
|:-----|:-----|
| `perf_opt/results/mtp_model/` | 好模型（w8a8-mtp）单服务全套：`_summary.json`、`prod_c{1,8,32,64}_4k.json`（MTP OFF 基线）、`mtpon_c{1,8,32}_4k.json`（MTP ON）、`base_c{1,32}_4k.json`（static_kernel 关）、`tp8x2_dp{0,1}_c32.json`（双服务 MTP OFF）、`gpqa_full_result.json`（141/198=71.21%）、`consistency.json` |
| `perf_opt/results/verify_1026/` | 2026-10-26 双服务+MTP 补测 4 点（`dualA/B_c8/c32.json`，聚合 1588.6 tok/s）＋显存/fp8 核实记录 |
| `perf_opt/results/clean_final/` | 2026-10-19 清洗会话 25 文件——在**已作废的坏模型**上测的，仅其"请求量效应"方法学结论仍有价值，绝对值不引用 |

**⚠️ 作废数据（绝对不要引用其绝对值）：**
阶段 0-5 早期所有结果（`results/matrix/`、`results/sched/`、`results/topo/`、`results/mtp/`、`results/phase5/` 大部分）都是在坏模型 w8a8-self 上测的——该模型输出退化（只会生成横杠），2026-10-23 查明并删除。这些数据中**同模型内的相对比较**（如调度开关的相对收益）仍可参考，但绝对值一律作废。文档里凡引用早期数据处均已标注。

---

## 4. 前同事的血泪教训（务必遵守）

1. **模型事故（最重要）**：自量化模型 w8a8-self 出生即坏（量化管线把 MTP 头也量化了+整体退化），项目曾全程误用它，直到 GPQA 复测得 2.5% 才发现。**教训：任何性能数据都必须配套一次文本质量/精度抽查；压测脚本用随机 token 输入，测不出内容好坏。** 好模型是官方量化版 w8a8-mtp（global_public 公共目录）与同方法重新量化的 w8a8-npu16（`models/DeepSeek-V4-Flash-w8a8-npu16`，280GB）。
2. **对比必须同模型、同会话、同请求量**：TTFT 对请求量极敏感（32req vs 128req 差 2.6 倍）；跨日/跨节点有 ~2× 残余差异。冻结口径：4K 输入/1K 输出/temp=0/seed=42/c1=16req、c8=64req、c32=128req、c64=128req；双服务每侧 64req。
3. **每个压测点必须做自洽校验**：wall_time ≈ 总输出 token ÷ 实测吞吐（偏差 >15% 即作废重跑）。曾有一批数据被并发压测污染（墙钟虚高 3.5-4×）靠此规则查出。
4. **显存读数**：npu-smi 的 `HBM-Usage(MB)` 才是真实显存（真实占用 ~59.8GB/卡=91%）；压测脚本曾因正则漏配读到 3.4GB 的错误值（**正则已修复**：`(\d+)\s*/\s*(\d+)`）。
5. **fp8 KV 已确认真正生效**（启动日志 `cache.py:269 Using fp8 data type to store kv cache`，KV 分配 14.13 GiB/卡）；其独立性能影响与 static_kernel 混在一起（组合 +1.3~1.7%），未单独隔离。
6. **MTP 结论**：好模型上 MTP=ON 是最大收益项（接受率 95.79%，吞吐 +45~75%），旧"MTP=OFF"结论作废。spec=2 因 cudagraph shape 冲突不支持。
7. **P99 才是 SLA**：c32 的 P99（8.0s）是 P50 的 9 倍，报告容量别只看 P50。
8. **只写实测数据**：缺失/未测项标 `null`/"未测"，不得补值、不得外推成结论（文档全部遵守此规则）。

---

## 5. 基础设施与操作流程（如何上远端干活）

**远端环境**：启智平台 notebook `quant-test`（工作空间 `昇腾卡公共空间`，16×910B2C，镜像 `glm53-w8a8-env:v2`）。CLI：`inspire`（本机 `~/.local/bin/inspire`，账号 `sii`）。
远端代码路径：`/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test`（共享盘，持久）。

**常用命令**：
```bash
inspire notebook status/start/stop quant-test --workspace 昇腾卡公共空间   # 实例管理
inspire notebook exec quant-test --workspace 昇腾卡公共空间 '<命令>'      # 执行远端命令
inspire notebook save-image quant-test --workspace 昇腾卡公共空间 -n <名> -v <版本> --wait  # 保存镜像
```

**远端操作的坑（都踩过）**：
- `exec` 单条命令约 2049 字符截断 → 传文件用 hex 编码 + <1600 字符分块：`python3 -c "open('PATH','ab').write(bytes.fromhex('HEX'))"` 逐块追加（远端无 xxd，base64 会损坏）
- 从 python subprocess 调 inspire 必须 `stdin=subprocess.DEVNULL`（否则被误判为 stdin 管道而拒绝）
- 后台任务必须 `setsid nohup ... &` 且先 mkdir 日志目录，否则进程随 exec 终端被回收
- 共享盘项目配额曾满（1.7T）→ 运行期产物写容器 `/tmp`，关键产物**每步立即 git commit 拉回本地**；实例停止/重启会清空 `/tmp`
- 服务器用完必须 `pkill -9 -f "VLLM::"`；实例用完按用户要求**关闭并核实 STOPPED**（用户曾因此不满：说关了没真关）
- 实例停止后首次重启服务器要重新图编译（~10-25 分钟，耐心等 health 200）

**代码同步**：本地 git push → 远端 `git fetch origin && git checkout origin/main -- <文件>`（远端共享盘配额限制，按文件同步；远端 repo 的 git reset --hard 可能因未跟踪文件失败）。

---

## 6. 最终配置速查（全部实测）

```bash
# 配置 A · 低延迟交互版（单组 TP8，8 卡，MTP=ON）—— c1 330ms/23.5ms/41.3 tok/s；c32 894ms/781.8 tok/s
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 perf_opt/scripts/start_server_phase5.py \
  --port 8000 --mtp 1 --static-kernel true --kv-cache-dtype fp8 \
  --cpu-binding true --dsa-cp true --multistream-overlap true

# 配置 B · 高吞吐批量版（双独立 TP8，16 卡，两台均 MTP=ON，客户端对半分流）—— c32 聚合 1588.6 tok/s（2.03×）
# 第二台同命令换 ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 与 --port 8001
```

启动器 `scripts/start_server_phase5.py` 已指向 `models/DeepSeek-V4-Flash-w8a8-npu16`。压测用 `scripts/bench_serving.py`；评测用 `scripts/gpqa_mtp.py`（sample_score 字段解析，勿用旧脚本）。

---

## 7. 重要资产

- **模型**：`models/DeepSeek-V4-Flash-w8a8-npu16`（280GB，2026-10-23 重新量化并验证）；量化脚本 `archive/dsv4_quant/quant_dsv4_npu16.py`，量化日志 `archive/dsv4_quant/requant_npu16_20261023.log`；坏模型取证元数据 `archive/dsv4_quant/forensics_w8a8_self/`（6.2MB，留作量化管线研究）
- **镜像**：`glm53-w8a8-env:v2`（2026-10-23 快照）
- **git**：main 分支，tag `perf-opt-final`（最新提交 9bcfe21）；全部结果与文档已入库

---

## 8. 遗留事项（可选后续，均为已标注的未测项）

1. c64 的 MTP=ON 单服务（只测到 c32）
2. 双服务 c64（MTP ON 与 OFF 都未测）
3. fp8 KV 与 static_kernel 的**独立**单变量隔离（现只有组合值 +1.3~1.7%）
4. cpu_binding / dsa_cp 关闭影响在好模型上的复测（好模型全程开着这两个开关，关闭后果只在坏模型上测过）
5. 真实共享前缀负载下的 prefix caching A/B（随机负载无收益不具代表性）
6. spec=2 解禁（cudagraph shape 冲突：spec+1=3 与 TP=8 序列并行不互质）
7. 更深的 KV 量化（当前只用 fp8）
8. 生产采样分布（temp>0、真实流量）下的 MTP 接受率
9. 修复后的 npu-smi 采样 regex 跑一次完整压测验证显存采样正确

---

## 9. 工作原则（前同事的约定，请保持）

- 所有结论必须有实测文件支撑；不编造、不补值、未测就标"未测"
- 每步结果立即 git commit + push（远端 /tmp 会丢，git 不会）
- 长任务自动值守：Monitor 心跳 + 关键事件通知，别让用户来问进度
- 重要结论变更要同步更新 `logbook.md`（最新在前）与 `PROGRESS.md`
- 用户重视：数据诚实 > 结论好看

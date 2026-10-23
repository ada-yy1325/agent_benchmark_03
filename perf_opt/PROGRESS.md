# 进度：阶段 0-4 完成，剩阶段 5/6
恢复规则：先读本文件，产物存在且校验通过的步骤跳过，从第一个 [ ] 继续。

## 阶段 0 · 基线
- [x] 0.1 环境核实建档        产物：env/env_info.json, env/paths_check.json
- [x] 0.2 建目录+本进度文件     产物：perf_opt/ 目录结构, PROGRESS.md
- [x] 0.3 压测脚本+冒烟 ✅     产物：scripts/bench_serving.py (含 NPU 采样 regex 修复), results/smoke/{smoke_c1,c2,c4}.json
  - smoke_c1: 2/2 ✅ TTFT=234.83ms TPOT=176.3ms fallback=False
  - smoke_c2: 4/4 ✅ TTFT=256.37ms TPOT=120.09ms
  - smoke_c4: 8/8 ✅ TTFT=283.99ms TPOT=367.94ms
- [x] 0.4 固化基线启动脚本     产物：scripts/start_server_baseline.py（git tag smoke-pass）
- [x] 0.5 启动基线服务+冒烟    产物：server_logs/baseline.log（tmux dsv4_baseline 运行中）
- [x] 0.6 基线压测 c=1 ×3      产物：results/baseline/baseline_c1_run{1,2,3}.json
  - run1: 32/32 ✅ wall=1522s TTFT p50=355.5ms TPOT p50=92.1ms
  - run2: 32/32 ✅ wall=1513s TTFT p50=354.0ms TPOT p50=91.8ms
  - run3: 32/32 ✅ wall=1511s TTFT p50=355.3ms TPOT p50=91.7ms
- [x] 0.7 基线汇总             产物：summary/baseline_summary.json
  - mean TTFT=347.6ms TPOT=92.0ms throughput≈10.8 out_tok/s (~21.6 actual)
  - 🔧 注意：基线 runs 使用 chunk 计数（~512/req），已修复为新版 `usage` 精确计数（phase 1 跑时会用新方法）
  - 🔧 /v1/completions 端点变更：服务可能被重启，新版代码已改用 `/v1/completions`
  - 📝 完整 Phase 0 报告已写入 logbook.md 顶部（2026-10-13），含环境差异、所有 bug 修复过程、基线数据、经验教训

## 阶段 1 · 性能画像
- [x] 1.1 矩阵规划（先跑 4K 档）产物：summary/matrix_plan（写在 findings.md 顶部）
- [x] 1.2 4K 档逐点压测 ✅       产物：results/matrix/4k_out1024_c*.json（7/7 全成功）
- [x] 1.3 2K 档逐点压测 ✅       产物：results/matrix/2k_out1024_c*.json（7/7 全成功）
- [x] 1.4 8K 档逐点压测 ✅       产物：results/matrix/8k_out1024_c*.json（6/6 全成功，~23 min）
- [x] 1.5 矩阵汇总 ✅           产物：summary/matrix_summary.{json,csv}（20 case，三档全量）
- [x] 1.6 分析+findings ✅       产物：findings.md（8K 数据 + 三档横向对比 + 瓶颈根因）
- [x] 1.7 git 提交+收尾 ✅        产物：git commit → GitHub（23 个结果 JSON + 3 个文档）

## 已知问题
- 🔴 远程 GitHub 不可达 → 使用 base64+gzip 分片同步文件到远程（注：2026-10-16 起远端 git fetch/pull 已恢复可用，可直接 git 同步）
- 🟡 CANN 25.0.rc1.1 (vs 预期 9.0.0) — 待确认影响
- ✅ NPU 显存采样 regex 已修复（npu-smi 的 HBM 行无 "MB" 后缀）
- 🟡 inspire exec 传输限制（2026-10-16 摸清，阶段 5/6 注意）：
  - `quant-test` 是受限 Notebook：无 SSH/SCP，exec 拒绝本地 stdin 管道
  - 单条 exec 命令约 2049 字符处会被截断 → 传文件用 **hex 编码 + <1600 字符分块**，
    `python3 -c "open('PATH','ab').write(bytes.fromhex('HEX'))"` 逐块追加（远端无 xxd，base64 会损坏）
  - 从 python subprocess 调 inspire 时必须 `stdin=subprocess.DEVNULL`
  - 后台任务要用 `setsid nohup ... &` 且日志重定向目录要先 mkdir，否则进程随 exec 终端被回收
- ✅ MTP 在此栈上强制 enforce_eager=True（eager=False 启动即崩），非可调项
- 🟡 **共享盘配额已满**（2026-10-19，项目目录 1.7T，主要在 models/）：远端写文件改用容器本地 /tmp；
  后续需清理 models/ 中间量化产物（w8a8-mtp/mtp-resume/npu16 等）——删数据需用户确认
- 🟡 **TTFT 测量依赖请求量**（2026-10-19 清洗会话确认）：32req vs 128req 的 c32 TTFT p50 差 2.6 倍；
  跨日/跨节点还有 ~2 倍残余差异。延迟对比必须固定请求量约定 + 同一会话背靠背

## 阶段 2 · 预填充调度优化（最高优先）
- [x] 2.0 测量核实（显存/拓扑/MTP）   产物：env/recheck.json
- [x] 2.1 chunked prefill 状态      产物：results/sched/chunk_status.md
- [x] 2.2 扫 max-num-batched-tokens  产物：results/sched/bat_scan/ (8/8, 最优=8192, 即是基线)
- [x] 2.3 扫 max-num-seqs           产物：results/sched/seqs_scan/ (clean, 最优=64 → c64 TTFT 62.7s→2.5s, 25× ↓)
- [x] 2.4 组合验证 c=32/48/64       产物：results/sched/final/ ⚠️ 该批数据被并发压测污染（wall 虚高 3.5-4×）
  - ✅ 2026-10-19 清洗会话重测（results/clean_final/，全部自洽）：c32(128req)=1972ms/569 tok/s,
    c48(128req)=2300.8ms/708 tok/s, c64(128req)=2292.7ms/965 tok/s —— 正式数字以此为准
  - 另发现 c32(32req)=5111.6ms：**TTFT p50 依赖请求量（2.6×）**，跨阶段延迟对比必须固定请求量
- [x] 2.5 调度最优配置小结          产物：summary/sched_best.json

## 阶段 3 · 并行拓扑（用满 16 卡）
- [x] 3.1 确认卡数/拓扑/DP 支持    产物：source code analysis; vllm-ascend 支持 native DP; EP=TP×DP
- [x] 3.2 方案C TP16              产物：scripts/run_topo_tp16.py → results/topo/tp16/
  - ❌ 失败：模型加载阶段 crash，`shape '[0, 1024, -1]' is invalid`（DeepSeek-V4 架构不兼容 TP16）
  - 日志：_server_tp16.log — HCCL 16 卡初始化成功但模型加载失败
- [x] 3.3 方案A/B DP2×TP8         产物：scripts/run_topo_dp2tp8.py → results/topo/dp2tp8/
  - ❌ 原生 DP 失败：vllm-ascend v0.23.0 单节点 DP 设备分配 bug (SetDevice error 507033)
  - ✅ 两独立 TP8 服务器方案成功：ASCEND_RT_VISIBLE_DEVICES 隔离，c1/c8/c32/c64 全跑完
  - 结论：唯一可行的 16 卡方案
- [x] 3.4 方案D DP4×TP4             ❌ 跳过：VRAM OOM (TP4 需 ~75 GB/卡 > 64 GB HBM)，实测加载即爆
- [x] 3.5 EP 专家并行核实           产物：probe_ep.py → EP 是附着在 TP/DP 上的开关，不是独立拓扑维度；所有配置已默认开启
- [x] 3.6 拓扑选型小结              产物：summary/topo_best.json
  - 结论：DeepSeek-V4-Flash-W8A8 在 16×NPU-910B2C 单节点上：
    - TP16：模型架构不兼容（shape error）
    - DP2×TP8 原生：vllm-ascend DP 设备分配 bug → 不可行
    - DP2×TP8 两服务器：唯一可行 16 卡方案，TP +27%（812.7 tok/s），TTFT 恶化
    - DP4×TP4：OOM 不可行
    - EP：已默认启用，非独立维度
    - 生产建议：TP8×2 独立服务（非 DP pooling），剩余 8 卡独立部署

## 阶段 4 · MTP 投机解码
- [x] 4.1 MTP 状态/接受率核实   产物：env/mtp_status.json, summary/mtp_acceptance.json
  - MTP 机制可用（spec=1），接受率：随机 token 4.35-4.73%，自然文本 16.64%（2026-10-16 审计补测）
  - spec=2 启动失败：cudagraph shape 冲突（spec+1=3 与 TP=8 序列并行不互质），**非量化问题**
  - MTP 在此栈上强制 enforce_eager=True（eager=False 启动即崩，12:08 日志为证）
- [x] 4.2 关/spec=1/更多 对比   产物：summary/mtp_comparison.json
  - OFF 比 ON 快 2.4%（c1）；spec=2 不可用（见 4.1）
- [x] 4.3 不同并发下净收益     产物：summary/mtp_best.json（含 2026-10-16 审计版）
  - off/1 × c1/c8/c32 六点齐全；c8 双侧重跑验证：ON TTFT 554-765ms vs OFF 1389-1421ms（方向稳定）
  - 全并发下 ON 的 TPOT/吞吐均差于 OFF（-2.3% ~ -10%）
  - 结论：生产用 MTP=OFF；若 c≈8 且重 TTFT 轻吞吐，MTP=ON 可选（TTFT 约减半，吞吐 -9%）
- [x] 4.4 enforce_eager 作用域验证 ✅ 已并入 4.1 审计（MTP 必须 eager，无调整空间）

## 阶段 5 · 其他开关 + 精度复测
- [x] 5.0-5.4 见历史记录（均在坏模型 w8a8-self 上完成，数据标注为历史数据）
- [x] 5.5 数值一致性 + 最终配置 GPQA 复测 ✅ 2026-10-23（好模型 w8a8-mtp）
  - 一致性：✅ 全过（temp=0 确定性正常）
  - GPQA：冒烟 3/3，**全量 141/198 = 71.21%**（与历史好模型 71.2-75.3% 吻合）
- [x] 🔴 模型事故处理 ✅ 2026-10-23
  - 查明：w8a8-self（自量化）出生即坏（输出横杠），性能项目全程误用；好模型 w8a8-mtp/npu16 曾存在后被删
  - 修复：切到 global_public 的 w8a8-mtp 完成全套复测（results/mtp_model/，全部自洽）
  - 重量化：quant_dsv4_npu16.py 原方法重跑成功（2h56m）→ models/DeepSeek-V4-Flash-w8a8-npu16（280GB）
  - 新模型验证：文本正常（EN/CN/MATH）+ GPQA 冒烟 3/3 ✅
  - 坏模型已删（293GB），6.2MB 量化元数据留档 archive/dsv4_quant/forensics_w8a8_self/
  - 启动脚本已切换：start_server_{mtp,param,phase5,isolated}.py → w8a8-npu16
- [x] 5.5b 好模型性能复测（结果见 results/mtp_model/ + logbook 2026-10-23 晚）
  - 生产配置（MTP off）：c1 319ms/23.6, c8 885ms/181.7, c32 1662ms/538.1, c64 1841ms/889.8
  - **MTP on：接受率 95.8%！吞吐 +45~75%，TTFT -46~53% → 生产建议 MTP=ON（结论反转）**
  - static_kernel 在好模型上仅 +1.3~1.7%；双服务器 1.99× 线性扩展

## 阶段 6 · 收尾
- [ ] 6.1 配置-性能总表（基线→各阶段→最终）           产物：summary/final_report_table.*
- [ ] 6.2 低延迟/高吞吐两套推荐配置（含 MTP 可选说明） 产物：summary/recommended_configs.md
- [ ] 6.3 /tmp 产物归集（git/共享盘）                  产物：git commit、目录核对
- [ ] 6.4 最终报告 + git tag 收尾                     产物：summary/final_report.md、tag
- [x] 6.5 实例镜像保存 + 关闭 ✅ 2026-10-23 22:30
  - 镜像：glm53-w8a8-env:v2（快照 322MB，READY）
  - 实例：quant-test 已 stop，status 核实 STOPPED
  - 量化日志归档：archive/dsv4_quant/requant_npu16_20261023.log

### 统一对照压测点（阶段 5 全程冻结）
| case | input | output | concurrency | num_requests |
|------|-------|--------|-------------|-------------|
| c1_4k  | 4096 | 1024 | 1  | 16  |
| c8_4k  | 4096 | 1024 | 8  | 64  |
| c32_4k | 4096 | 1024 | 32 | 128 |
| c64_4k | 4096 | 1024 | 64 | 128 |

### 基础设施
- 灵活启动脚本：`scripts/start_server_phase5.py`（支持 --mtp/--prefix-caching/--graph-mode/--cpu-binding 等全部开关）
- 编排脚本：`scripts/run_phase5.py`（--step 5.0/5.1/5.2/5.3，顺序执行+自洽校验，结果写 /tmp/phase5/）
- 压测脚本：`scripts/bench_serving.py`（沿用，含 NPU 采样 regex 修复 + usage 精确计数）
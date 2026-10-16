# 进度：阶段 0 + 1
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
- 🔴 远程 GitHub 不可达 → 使用 base64+gzip 分片同步文件到远程
- 🟡 CANN 25.0.rc1.1 (vs 预期 9.0.0) — 待确认影响
- ✅ NPU 显存采样 regex 已修复（npu-smi 的 HBM 行无 "MB" 后缀）

## 阶段 2 · 预填充调度优化（最高优先）
- [x] 2.0 测量核实（显存/拓扑/MTP）   产物：env/recheck.json
- [x] 2.1 chunked prefill 状态      产物：results/sched/chunk_status.md
- [x] 2.2 扫 max-num-batched-tokens  产物：results/sched/bat_scan/ (8/8, 最优=8192, 即是基线)
- [x] 2.3 扫 max-num-seqs           产物：results/sched/seqs_scan/ (8/8, 最优=64→TTFT 2.5s, 25× ↑)
- [x] 2.4 组合验证 c=32/48/64       产物：results/sched/final/ (c32=957ms, c48=12.6s, c64=13.9s; prefill瓶颈确认)
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
- [ ] 4.1 MTP 状态/接受率核实
- [ ] 4.2 关/spec=1/更多 对比
- [ ] 4.3 不同并发下净收益

## 阶段 5 · 其他开关 + 精度复测
- [ ] 5.1 prefix caching 对比
- [ ] 5.2 cudagraph / eager 对比
- [ ] 5.3 additional config 逐项
- [ ] 5.4 KV cache 量化
- [ ] 5.5 GPQA 精度复测

## 阶段 6 · 收尾
- [ ] 6.1 配置-性能总表
- [ ] 6.2 低延迟/高吞吐推荐配置
- [ ] 6.3 最终报告 + git 收尾
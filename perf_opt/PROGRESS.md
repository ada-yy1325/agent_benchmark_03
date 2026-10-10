# 进度：阶段 0 + 1
恢复规则：先读本文件，产物存在且校验通过的步骤跳过，从第一个 [ ] 继续。

## 阶段 0 · 基线
- [x] 0.1 环境核实建档        产物：env/env_info.json, env/paths_check.json
- [x] 0.2 建目录+本进度文件     产物：perf_opt/ 目录结构, PROGRESS.md
- [ ] 0.3 压测脚本+冒烟        产物：scripts/bench_serving.py, results/smoke/
- [ ] 0.4 固化基线启动脚本     产物：scripts/start_server_baseline.py（git tag baseline）
- [ ] 0.5 启动基线服务+冒烟    产物：server_logs/baseline.log
- [ ] 0.6 基线压测 c=1 ×3      产物：results/baseline/baseline_c1_run{1,2,3}.json
- [ ] 0.7 基线汇总             产物：summary/baseline_summary.json

## 阶段 1 · 性能画像
- [ ] 1.1 矩阵规划（先跑 4K 档）产物：summary/matrix_plan（写在 findings.md 顶部）
- [ ] 1.2 4K 档逐点压测        产物：results/matrix/4k_out1024_c*.json（成功的跳过）
- [ ] 1.3 2K 档逐点压测        产物：results/matrix/2k_out1024_c*.json
- [ ] 1.4 8K 档逐点压测        产物：results/matrix/8k_out1024_c*.json
- [ ] 1.5 矩阵汇总             产物：summary/matrix_summary.{json,csv}
- [ ] 1.6 分析+findings        产物：summary/findings.md（甜蜜点/劣化点/瓶颈）
- [ ] 1.7 git 提交+收尾        产物：git commit，更新 PROGRESS
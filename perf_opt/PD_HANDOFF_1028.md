# PD 分离任务交接说明（2026-10-28，给下一位 Agent）

## 一句话现状

目标是给 DeepSeek-V4-Flash（W8A8 量化，昇腾 910B2C ×16）做 **PD（prefill/decode）分离**以改善 TTFT 和 output tok/s。所有路径已探明，补丁已写好，但**实测被平台 NPU 环境回归挡住**——服务器启动全部失败。本文说明卡点、证据和恢复步骤。

## 环境（全部事实）

- 唯一工作实例：`quant-test`（工作空间 `昇腾卡公共空间`，16×910B2C，镜像 `glm53-w8a8-env:v2`）。**不要创建其他实例。**
- 好模型：`models/DeepSeek-V4-Flash-w8a8-npu16`（280GB，重新量化并验证：文本正常、GPQA 冒烟 3/3）
- 对比基准（最优现状）：双服务 MTP=ON，聚合 **1588.6 tok/s @c32**（单机 MTP=ON 781.8；统一服务器 c32 TTFT 4114ms / 784.3 tok/s）
- CLI：`inspire notebook exec quant-test --workspace 昇腾卡公共空间 '<命令>'`；上传文件用 hex 分块（<1600 字符/块，`bytes.fromhex` 追加）；后台任务用 `setsid nohup ... &`；远端代码在 `/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test`，git 已同步（最新 commit 580949a）

## PD 三条路径的结论（已测明）

| 路径 | 结论 | 证据 |
|:-----|:-----|:-----|
| ① vLLM 原生 DP=2 单部署 | ❌ 不可行 | `AssertionError: expected OutputCode, got _CompiledFxGraph / GraphModuleImpl`（两种编译配置均复现）——vllm-ascend 编译管线不支持 DP>1 |
| ② 两独立实例 + AscendStore ShmBackend KV 池 | ⚠️ 短 prompt 可用（≤512 token 全跳过，1.5-2× 加速）；长 prompt 只命中首段 ~512 token | 两阶段测试（`test_pd_two_phase.py`）复现 |
| ③ 客户端流水线编排 | 依赖 ② 的命中率；512-token 命中下 4K 无收益 | `pd_pipeline_bench.py` 32 请求实测 |

## 根因与已写的补丁

**根因**：vLLM V1 逐步填充 `request.block_hashes`（每个调度步填一段）；consumer 首次调度时只有首 chunk（4 blocks = 512 token）的哈希，AscendStore 池的 lookup（`pool_scheduler.get_num_new_matched_tokens` → `client.lookup(token_len, request.block_hashes, ...)`）因此只能匹配首段。

**补丁**（已完成、未实测）：`perf_opt/scripts/pd_full_hash_fix.py` —— 在 lookup 前用 vLLM 自己的链式哈希 `hash_block_tokens` 把 block_hashes 补全到整个 prompt（同环境同函数，键与 producer 保存的一致）。幂等文件补丁，带 `.pd_bak` 备份、py_compile 校验、失败自动回滚。修复过一次 import 锚点问题。

**已验证的必要条件**：
- token-id prompt 完全不命中池（必须文本 prompt，走 chat/completions）
- `--enable-prefix-caching` 必须开（否则 block_hashes 不完整）——已加入 `start_pd_v2.py`
- `kv_buffer_size` 字段不控制实际分配（/dev/shm psm 实测固定 ~500MB）——池容量上限待查，但对"首段命中"问题无影响
- producer/consumer 串行启动（CANN 编译竞态）、consumer 跳过 LookupKeyServer（`pd_fix_connector.py`）

## 卡点：平台 NPU 环境回归（2026-10-27 下午起）

**现象**：任何服务器启动都在 worker 的 `profile_run` 阶段崩：
```
RuntimeError: fill_: aclnnInplaceFillScalar failed, error code is 561103
FillAiCore ADD_TO_LAUNCHER_LIST_AICORE failed
```
**证据链**（排除法）：
- 10-27 上午（同镜像同脚本）启动正常；10-26 正常 → 下午出现的增量变化
- 三个节点（412/416/420）全部复现 → 非节点问题（412 另有节点级异常，已避开）
- **纯标准配置**（无 PD 配置、无补丁、`start_server_phase5.py --mtp 0`）同样崩溃 → 与我们的代码无关
- 关 static_kernel、重启实例清 NPU 状态 → 无效
- 判定：平台驱动/固件更新导致 vllm-ascend 的 ATB/npugraph 图编译路径失效

**尚未尝试的绕过**：`--enforce-eager`（纯 eager、完全绕过图编译）——2026-10-28 正在 quant-test（node 416）上测试。若 eager 能启动，则：a) 平台回归只影响图编译路径；b) 可用 eager 模式完成 PD 机制验证（PD 的 KV 传输与图模式无关，eager 慢但能证明机制与量化收益）。

## 平台修复后的恢复步骤（按序执行，预计 1 小时内出结果）

1. `python3 perf_opt/scripts/pd_full_hash_fix.py`（打补丁，幂等）
2. `python3 perf_opt/scripts/pd_fix_connector.py`（connector 修复）
3. `setsid nohup python3 perf_opt/scripts/start_pd_v2.py --serial > /tmp/pd_test/_v2.log 2>&1 < /dev/null &`（串行启动双服务器）
4. 双健康后跑两阶段 4K 验证：producer 发 4K 文本 prompt（max_tokens=64）→ consumer 同 prompt（max_tokens=1024）→ 看 consumer 日志 `prompt throughput` 是否 ≈0（补丁生效=全量命中）
5. 生效则跑流水线基准：`python3 /tmp/pd_pipeline_bench.py --num-requests 32 --consumer-concurrency 32 --lookahead 4`（若 /tmp 已清，用仓库 `perf_opt/scripts/pd_pipeline_bench.py` 重传）
6. 对比基准：TTFT 应远低于 4114ms（prefill 与 decode 解耦）；吞吐应 ≥781.8（consumer 专职 decode），双服务版本预期 ≥1588.6

## 给平台报障的要点

错误签名：`aclnnInplaceFillScalar failed, error code is 561103`（vllm_ascend worker profile_run 阶段）；时间：2026-10-27 下午起；影响：所有 vLLM 服务器无法启动。logbook.md 有完整时间线。

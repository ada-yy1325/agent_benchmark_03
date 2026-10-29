# PD 分离任务交接书（2026-10-29，给下一位 Agent）

> 前情：本项目是昇腾 910B2C ×16 上 DeepSeek-V4-Flash（W8A8）推理性能优化。阶段 0-6 已完成（成绩单见 `summary/full_optimization_summary.md`，最优基线：双服务 MTP=ON 聚合 1588.6 tok/s @c32）。本任务 = PD（prefill/decode）分离的最后一公里。

## 一、当前状态（10-28 晚收尾时的真实状态）

- **quant-test 实例已关闭**（status 复核 = STOPPED）。
- **镜像已保存**：`glm53-w8a8-env-pd1028:v1`（SUCCESS，私有）。包含：静态内核崩溃修复（config 清空+动态回退）、启动脚本修复、两个正式站点包补丁。**不含**：调试探针（已清除）。
- PD 双服务（8000 prefill / 8001 decode）**可稳定启动**，零报错。
- **KV 池端到端传输未打通**。三个原因已全部定位（探针矩阵实测）：
  1. **16384 token 粒度闸门**：DSV4 MLA 压缩组（ratio 128）→ `cache_transfer_granularity=16384`；<16K token 的 prompt 在 `KVPoolScheduler.get_num_new_matched_tokens` 开头被直接 return，永不 lookup。
  2. **producer 保存路径未触发**：`AscendStoreConnector.save_kv_layer` 开头 `if not self.use_layerwise: return`；非 layerwise 模式下无任何保存执行。17.5K prompt 实测：lookup 执行（2190 哈希）→ hit=0（池空）。
  3. kv_buffer_size 配置不生效（/dev/shm psm 恒 ~251MB；17.5K 请求 KV ≈250MB 已贴容量线）。

## 二、任务清单（按顺序执行，每完成一项汇报一次）

### 任务 1：恢复环境（10 分钟）
1. `inspire notebook start quant-test --workspace 昇腾卡公共空间`，等 status 显示 RUNNING。
2. 验证三件事（一条 exec 搞定）：
   - fill 复活：`ASCEND_RT_VISIBLE_DEVICES=0 python3 -c "import torch, torch_npu; t=torch.zeros(8, device='npu'); t.fill_(1.0); torch.npu.synchronize(); print('FILL OK', t.sum().item())"`
   - 图捕获复活：`NPUGraph` 迷你捕获测试（含 fill 记录+重放，见 PD_HANDOFF_1028.md 第 2 节的方法）
   - 两个补丁仍在：`grep -c "PD full-hash fix" <pool_scheduler.py>; grep -c "not is_consumer" <ascend_store_connector.py>`（路径见下文第四节）
   - 若补丁丢失（镜像重建过）：重跑 `python3 perf_opt/scripts/pd_full_hash_fix.py` 和 `pd_fix_connector.py`（幂等）。

### 任务 2：修复保存路径（核心任务，预计 1-2 小时）
1. 远端 `perf_opt/scripts/start_pd_v2.py` 的 `KV_TRANSFER_CONFIG["kv_connector_extra_config"]` 加 `"use_layerwise": true`（**producer 和 consumer 共用同一字典，改一处即两侧生效**；注意 layerwise 模式对性能有影响，先验证机制再谈性能）。
2. 重启双服务（串行启动见 PD_HANDOFF_1028.md 第 3 节）。
3. **用 ≥16384 token 的 prompt 跑两阶段验证**：远端 `/tmp/pd4k.py` 已配置为 BASE×140 ≈ 17.5K token（stop/start 不重置 /tmp；若丢失，仓库 `perf_opt/scripts/pd_4k_two_phase.py` 的 PROMPT 改 `BASE * 140` 后 hex 分块上传）。
4. 判定：consumer 日志里 10 秒窗口 `Avg prompt throughput ≈ 0` = 池加载成功；TTFT 应远小于 producer 的 17K prefill（~5s）。若 hit 仍为 0：临时在 `pool_worker.lookup_scheduler` 的 debug 行后插 INFO 探针（用完删除），查 exists 计数判断是保存没发生还是哈希对不上。

### 任务 3：并发流水线验证（1 小时）
- 用 17K+ prompt 版本跑 `pd_pipeline_bench.py`（32 请求，注意把脚本里 `TARGET_INPUT_TOKENS` 改 17000；--post-delay 参数已存在）。
- 成功标准：consumer 不再全量本地重算（日志 prompt throughput ≈ 0），TTFT p50 明显低于统一服务器 4114ms。
- 池容量红线：251MB ≈ 1 个 17K 请求的 KV。并发 >2 时必然互相挤掉——若命中率随并发暴跌，这是已知原因；可尝试在代码里找 psm 大小硬编码（`pool_worker.py` 中 buffer 分配处）调大，或接受低并发结论。

### 任务 4：性能恢复（可选但推荐，1-2 小时）
- 静态内核再生：`start_pd_v2.py` 改回 `enable_static_kernel:true`（保留图模式），启动后 torch_npu 的 `compile_static_kernel` 会自动重新编译全部内核（约 12 分钟，产物会回填 `opp/static_kernel/ai_core/` 并更新 config.ini/binary_info_config.json）。
- 成功后重跑任务 2/3 的验证 + 与 1588.6 tok/s 基线对比，出最终成绩。
- **失败则回滚**：恢复 `config.ini`（清空 load_priority）+ `binary_info_config.json`（`{}`）的 .bak，`enable_static_kernel:false` 重新启动即可回到动态模式。

### 任务 5：收尾（同 10-28 流程）
- 清探针 → 文档更新（logbook 置顶、PD_HANDOFF/ISSUE_REPORT/成绩单）→ git 提交（本地+远端，远端用路径限定 `git add perf_opt/scripts/...`，**禁止 `git add -A`**——会扫 280GB 模型目录卡死）→ 保存镜像 → 关闭实例 → **status 复核 STOPPED** → 邮件汇报。

## 三、工作文件在哪里

| 文件 | 位置 | 内容 |
|:-----|:-----|:-----|
| **PD 操作手册（先读这个）** | 本地 `perf_opt/PD_HANDOFF_1028.md` | 环境事实、静态内核陷阱、启动/验证步骤、三因修复、实测数据 |
| 问题报告 | 本地 `perf_opt/PD_ISSUE_REPORT_1028.md` | 根因定案+证据链、探针结论、给下一位的问题 |
| 成绩单 | 本地 `perf_opt/summary/full_optimization_summary.md` 第六章 | PD 实验全部数据与诚实结论 |
| 过程记录 | 本地 `logbook.md`（顶部为 10-28 最新） | 每天做了什么/问题/解法 |
| 启动器 | 本地+远端 `perf_opt/scripts/start_pd_v2.py` | 远端版本 = 本地 HEAD（已同步提交 ca4697e） |
| 补丁脚本（幂等） | 远端仓库 `perf_opt/scripts/pd_full_hash_fix.py`、`pd_fix_connector.py` | 镜像重建后需重跑 |
| 验证客户端 | 本地 `perf_opt/scripts/pd_4k_two_phase.py`（远端 /tmp/pd4k.py 是 17K 版）；`pd_pipeline_bench.py`（有 --post-delay） | hex 分块上传方法见手册第 1 节 |
| 模型 | 远端 `/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-npu16` | 280GB 好模型，别动 |
| 站点包补丁位置 | `/opt/mamba/lib/python3.13/site-packages/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`（full-hash）、`ascend_store_connector.py`（consumer 不建 LookupKeyServer） | 检查/插桩都在这里 |
| 静态内核配置 | `/usr/local/Ascend/cann-9.0.0/opp/static_kernel/ai_core/config.ini`（.bak 在旁）、`config/ascend910b/binary_info_config.json`（.bak 在旁） | **绝对禁止删 static_kernel_* 目录** |

## 四、给 Agent 自己的执行纪律（防卡死，务必遵守）

1. **一次只做一个实验**：启动/等待/验证/汇报完成后再做下一个。禁止同时跑多个后台实验（CANN 编译竞态 + 资源互踩是历史卡点）。
2. **每步设超时与放弃条件**：例如"等 10 分钟双 200 没来就查日志并停下汇报，不要无限重试"。
3. **同一命令失败两次必须停下换思路**，禁止第三次原样重试（历史教训：FINAL1-5 反复崩就是没有停下来挖根因）。
4. **长任务先落日志再后台跑**：一律 `setsid nohup python3 -u ... > /tmp/pd_test/xxx.log 2>&1 < /dev/null &`，然后轮询日志关键字，而不是阻塞等待。
5. **所有 exec 命令避免交互**：不加 `-i`、不用 `less/tmux attach` 查看，用 `tail/grep`。
6. **不新建实例、不动模型目录、不 `git add -A`**（远端）。
7. **每完成一个任务更新 `logbook.md` 与本地 git 提交一次**，崩溃也能从检查点续。
8. 收尾必做：清探针 → 保存镜像 → 关实例 → status 复核 STOPPED。

## 五、上一位的推测与未验证点（供参考，非结论）

- `use_layerwise:true` 是让 producer 执行逐层保存的最直接开关，但 layerwise 模式的 NPU 拷贝开销未知，机制验证通过后建议关回非 layerwise 并另找保存触发点（事件驱动路径）。
- 哈希链一致性未实测（历史 512-token 问题已由 full-hash 补丁覆盖，但补丁按 128 粒度、pool_worker 的 hash_block_size=8 的差异是否影响键匹配，需在任务 2 的探针里顺带确认）。

## 六、下一位 Agent 提问的补充回答（2026-10-29）

1. **use_layerwise 修改**：确认——`build_args` 里 `cfg = dict(KV_TRANSFER_CONFIG)` 是浅拷贝，`kv_connector_extra_config` 内层字典两侧共享，改 KV_TRANSFER_CONFIG 一处即两侧生效，无需额外字段。注意两个细节：① 内层字典被共享且 `build_args` 会对它 `["consumer_is_to_load"]=True`（这个泄漏到 producer 侧无害，别修它）；② 若将来要加**单侧**字段，必须先 `cfg["kv_connector_extra_config"] = dict(...)` 深拷内层。另：`use_layerwise:true` 会让 `requires_piecewise_for_cudagraph` 返回 True（框架自动切换 piecewise 图捕获），启动行为会变，属预期。**保存是否真的触发别靠猜**：`vllm_ascend/worker/model_runner_v1.py` 里 grep 不到 save_kv_layer 的调用（调用在 vllm 基础 model_runner 里），首次启动后在 `save_kv_layer` 入口插一行 INFO 探针确认，1 分钟的事。
2. **任务 4 时机**：打通任务 2/3 后**立即做任务 4**（静态内核再生），原因：基线 1588.6 tok/s 是静态内核环境测的，动态内核慢 4-5 倍，不恢复静态内核做的性能对比没有意义；且任务 4 失败回滚只需恢复两个 .bak 文件。顺序：2/3 打通 → 4 静态内核 → 用静态内核重验 2/3 出最终成绩 → 5 收尾。若任务 4 反复失败（编译 ~12 分钟超时或崩），就以动态内核数据收尾并在文档里明示口径。
3. **PROGRESS.md**：不要动老的 `perf_opt/PROGRESS.md`（阶段 0-5 历史）。新建 `perf_opt/PD_PROGRESS.md` 做 PD 专项检查点，与其它 PD 文档放在一起；每完成一个子任务追加一行（任务+结果+下一步）并随 git 提交，卡死重启后从它续跑。

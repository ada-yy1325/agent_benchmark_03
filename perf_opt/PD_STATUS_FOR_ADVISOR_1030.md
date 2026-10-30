# PD 分离现状与求助问题（2026-10-30，给顾问 Agent）

## 0. 任务背景

在启智平台 quant-test 实例（16× Ascend 910B2C 64GB，节点 infra-gpu-npu-478）上，给 DeepSeek-V4-Flash（W8A8 量化，280GB，MLA 架构）做 **PD（prefill/decode）分离**：两个独立 vLLM 实例（prefill 8000 端口卡 0-7 / decode 8001 端口卡 8-15，均 TP8+EP8），通过 **AscendStoreConnector**（backend=shm，kv_buffer_size 配置 8e9 但 /dev/shm 池实测恒 ~251MB）+ LookupKeyServer（zmq IPC，socket 文件 /tmp/lookup_rpc_port_14579_dp_rank0）共享 KV。目标：consumer 跳过本地 prefill、TTFT 下降、decode 吞吐提升。
环境：vLLM 0.23.0 + vllm-ascend；CANN 9.0.0；torch_npu 2.11.0.rc4；Python 3.13。基线：双服务 MTP=ON 聚合 1588.6 tok/s @c32；统一服务器 c32 TTFT 4114ms。

## 1. 已完成的真实成果（非虚）

1. **历史崩溃总根源已修**：启动脚本误删 `opp/static_kernel/static_kernel_*` 预编译内核包（平台不自动再生）→ 所有 fill 算子崩 561103。修复：`config.ini` 清空 load_priority + `binary_info_config.json` 置 `{}`（动态 JIT 回退）+ 清理命令无害化 + `enable_static_kernel:false`（保留 npugraph 图模式）。**此后 PD 双服务每次启动都稳定成功、零报错**（FINAL6/7/9/10/11 均验证）。
2. 环境随实例换节点（478）重建后，静态内核包与配置**全量恢复**（原 8 包 + 10-29 新增 4 包），环境当前处于完全健康状态。
3. 两个站点包补丁在岗：full-hash（把 consumer 查池的 block_hashes 补全到整个 prompt）、pd_fix_connector（consumer 不自建 LookupKeyServer，连 producer 的）。
4. 已排除的路线：vLLM 原生 DP=2（编译管线断言失败）；layerwise 保存（官方报错 "AscendStore layerwise mode does not yet support hybrid KV cache groups"——DSV4 的 6 个混合 KV 组不兼容）；Mooncake（NPU 引擎 TCP 不可用 + 历史崩溃）。

## 2. 当前问题：KV 池"存→查→取"链路三个断点（全部探针实测）

### 断点① 16384 token 粒度闸门（已确认，机制如此）
`KVPoolScheduler.get_num_new_matched_tokens` 开头：
```python
if token_len < self.cache_transfer_granularity: return 0, False
```
`cache_transfer_granularity = lcm(128, 各组 128×compress_ratio)`，DSV4 MLA 的 c128 压缩组 → **16384**。`discard_partial_chunks=True` 时先向下取整。探针实测：2772/4066 token 的 prompt 全部被挡，永不 lookup；17518 token 的 prompt floor 到 16384 恰好过闸门。
**影响**：<16K token 的 prompt 完全不会用池（我们基准协议是 4K，直接全灭）。

### 断点② producer 保存不完整（已确认，疑似上游 bug）
非 layerwise 模式的保存链路：核心调度器 `build_connector_meta` → worker `start_load_kv`/`wait_for_save` → `kv_send_thread` 把 can_save 请求的块写入池。
探针实测（17.5K prompt 请求全周期 69 次调度步）：
- `PDDBGC build_meta requests=1` 仅出现 **1 次**（第一步，新请求 chunk=8192 token），其余 68 步 requests=0；
- 第 2/3 步（chunked prefill 续块）走 `scheduled_cached_reqs` 分支，其中 `if not new_block_ids: continue`——**续块复用已分配块、无新块 → 被跳过，第 2/3 块的保存元数据从未生成**；
- worker 侧确认收到过那唯一 1 个请求（PDDBGD requests=1 ×8 workers），wait_for_save 也见过一次（PDDBGA reqs=1 ×8）。第一块是否真的被 kv_send_thread 写进池**未验证**（发送线程无日志）。
**影响**：即使查询打通，池里最多只有第一块 8192 token 的 KV，且 17518 token 的请求需要 16384 起查——连"半截命中"都凑不齐。

### 断点③ consumer 查询没有到达 producer 的应答服务（已确认现象，原因不明 ⚠️ 求助重点）
- 10-28 探针：consumer 的 `client.lookup(token_len=16384, 2190 个哈希)` **执行了并返回 hit=0**（PDDBG1 打印在 lookup 返回之后）；
- producer 侧 `lookup_scheduler` 入口探针（今天插在门控过滤**之前**）**零输出**——producer 从未处理任何查询；
- 但 consumer 没挂起（zmq REQ/REP 无超时，若无人应答 recv 会永久阻塞），请求正常完成 → consumer 拿到了一个"0"的回复；
- socket 文件存在（srwxr-xr-x /tmp/lookup_rpc_port_14579_dp_rank0，producer 启动时创建）；
- consumer 全量本地重算 17518 token（日志 prompt throughput 1751.6 tok/s ×10s 窗口 ≈ 全部 prompt）。
**疑点候选**（未验证）：① LookupKeyServer 的线程是否已死（process_request 内 lookup_scheduler 若抛异常，线程死亡但 socket 仍在，之后谁在应答？）；② 同一路径被多个实例绑定（producer 的 EngineCore connector 与 TP0 worker connector 的 parallel_config.rank 都是 0，各自试图 bind 同一 socket，第二个 bind 的失败是否导致某个 server 线程状态异常）；③ zmq 连接方向/命名空间问题（同容器同 /tmp，理论上不存在）。**尚无证据支持任一候选——需要顾问判断还该查什么。**

## 3. 关键文件与复现

- 启动器：`/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/perf_opt/scripts/start_pd_v2.py`（git 已提交；--serial 串行启动）
- 站点包（探针目前仍在，重启后生效）：
  - `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`：full-hash 补丁 + PDDBGC（build_meta 计数）
  - `.../ascend_store/pool_worker.py`：PDDBGA（wait_for_save 计数）、PDDBGB（lookup_scheduler 入口）、PDDBGD（start_load_kv 计数）
  - `.../ascend_store/ascend_store_connector.py`：pd_fix_connector 补丁（is_consumer 跳过建 server）
- 复现：启动双服务 → `python3 /tmp/pd17k.py`（17.5K token 两阶段客户端）→ 查 `/tmp/pd_test/prefill_v2.log`（producer 侧探针）与 `decode_v2.log`
- 结果日志：`/tmp/pd_test/_17k*.log`、`_final*.log`

## 4. 请求顾问回答的问题（按优先级）

1. **断点③**：同容器内 zmq IPC REQ/REP，consumer 的 lookup 返回 0 而 producer 的 lookup_scheduler 从未执行，consumer 也不阻塞——最可能的机制是什么？还需要什么证据才能判定？（建议具体到检查项：LookupKeyServer 线程存活、socket 绑定冲突、engine_id 相关路径差异、zmq linger/序列号等）
2. **断点②**：chunked prefill 续块时 `scheduled_cached_reqs` 的 `new_block_ids` 为空导致保存元数据不生成——这是 vllm-ascend 上游 bug 还是配置问题？最小修复建议（配置侧：producer 关 chunked prefill / 调大 max-num-batched-tokens 使 17.5K 一次调度？代码侧：cached_reqs 分支按 token_len 增量生成 ReqMeta 而不依赖 new_block_ids？）
3. **断点①**：16384 粒度闸门对 4K 基准协议是硬伤。绕过是否可行（discard_partial_chunks=false + 调 granularity）还是 MLA c128 族的池布局根本要求？若不可行，PD 路线对 <16K 工作负载是否应该直接判死？
4. 综合判断：在当前 vllm-ascend 0.23.0 构建下（layerwise 不支持 hybrid、DP=2 编译断、Mooncake NPU 不可用），PD 是否还有可行路线？还是应升级 vllm-ascend 版本或改用 MindIE 等官方方案？给出你的推荐和理由。

## 5. 顾问裁决（2026-10-30 收到，已采纳执行）

**核心结论：AscendStore 对 DSV4 在 vllm-ascend 0.23.0 必败，立即停止；官方正道 = 官方镜像 + MooncakeHybridConnector。**
- 断点③：判断为残留/重复 REP 进程应答（查清也救不了①②，不再追查）
- 断点②：0.23.0 非 layerwise 保存链路的上游缺陷（配置侧绕过显存压力大、代码侧高风险，不做）
- 断点①：16384 粒度是 MLA c128 物理布局要求，4K 负载在 AscendStore 路线直接判死
- Mooncake 历史排除理由作废：NPU 走 AscendDirect 而非 TCP；历史崩溃根因（静态内核误删）已修复
- 执行：官方镜像 `m.daocloud.io/quay.io/ascend/vllm-ascend:DeepSeekV4-flash-0731`；先单机 1P1D 探索（P 0-7 / D 8-15，MooncakeHybridConnector，--no-disable-hybrid-kv-cache-manager，extra config prefill/decode dp_size=1 tp_size=8，P eager+MTP=1，D FULL_DECODE_ONLY，kv_port 避开 20000-36000）；≤2 次失败即止；MindIE 留后续
- 预期管理：PD 价值在 TTFT p99 长尾与 TPOT 抖动，稳态吞吐仍用配置 B

## 6. 执行进度（10-30 实时）

1. ✅ AscendStore 双服务已清理，探针停止追加
2. ✅ 官方镜像拉取已启动（daocloud 前 1.7GB @1.4MB/s 后全线路被平台限速至 ~1KB/s，改 quay 直连同样被限——**实例整体出网限速，拉取挂起等恢复**）
3. ✅ 容器可行性已验证：docker（podman）可用 + 16 davinci 设备直通可见
4. ⏳ 等网络恢复 → 镜像就位 → 单机 1P1D 探索（P 0-7 / D 8-15）
5. 备选：平台公共镜像 `vllm-ascend:v0.30.0rc2`（需实例重建）；旧环境可随时用镜像 pd1028 回滚

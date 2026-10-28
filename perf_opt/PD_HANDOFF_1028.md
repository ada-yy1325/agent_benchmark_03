# PD 分离完整操作手册（2026-10-28 终版，含全部踩坑）

> 目标：DeepSeek-V4-Flash（W8A8 量化，昇腾 910B2C ×16）PD（prefill/decode）分离。
> 当前状态：**双服务可稳定启动**；KV 池端到端传输**未打通**（闸门与保存路径已定位，见第 5 节）。

## 1. 环境（全部事实）

- 唯一工作实例：`quant-test`（工作空间 `昇腾卡公共空间`，16×910B2C 每卡 64GB，镜像 glm53-w8a8-env:v2）。**不要创建其他实例。**
- 好模型：`models/DeepSeek-V4-Flash-w8a8-npu16`（280GB，重新量化已验证：文本正常、GPQA 冒烟 3/3）
- vLLM 0.23.0 + vllm-ascend；CANN 9.0.0；torch_npu 2.11.0.rc4；远端代码 `/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test`（git main）
- CLI：`inspire notebook exec quant-test --workspace 昇腾卡公共空间 '<命令>'`；上传文件用 hex 分块（<1500 字符/块，`bytes.fromhex` 追加，stdin=DEVNULL）；后台任务 `setsid nohup python3 -u ... &`
- 对比基准（最优现状）：双服务 MTP=ON 聚合 1588.6 tok/s @c32；统一服务器 c32 TTFT 4114ms

## 2. ⚠️ 静态内核陷阱（所有历史崩溃的总根源）

`opp/static_kernel/` 下的 8 个 `static_kernel_*` 目录是镜像构建时用 bisheng-compiler 预编译的内核包，**平台运行时不会再生**。一旦删除，任何 aclnn 算子走静态路径（图捕获的 fill 必走）就崩 561103 / FillAiCore，且不会回退动态编译。

- **绝对禁止**：任何脚本执行 `rm -rf opp/static_kernel/static_kernel_*`（start_pd_v2.py/v3.py 的清理步骤已改为无害 `ls`）
- 当前容器状态：内核包已被历史误删；`config.ini`（load_priority 已清空，.bak 在旁）与 `binary_info_config.json`（已置 `{}`，.bak 在旁）使运行时走动态 JIT 回退——**fill/图捕获均正常**（真机验证：独立 fill OK、迷你图捕获 OK）
- 若将来镜像重建/内核恢复，静态内核路径可自动重新生效（`enable_static_kernel:true` 亦可触发 torch_npu 的 compile_static_kernel 重新编译，约 12 分钟）

## 3. 启动 PD 双服务（已验证可用的完整步骤）

```bash
cd /inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test
# 1) 站点包补丁（幂等）：full-hash + connector 修复
python3 perf_opt/scripts/pd_full_hash_fix.py     # 补全 block_hashes 到整个 prompt
python3 perf_opt/scripts/pd_fix_connector.py     # consumer 不建 LookupKeyServer，连 producer 的
# 2) 串行启动（decode 先起，prefill 后起；避免 CANN 编译竞态）
setsid nohup python3 -u perf_opt/scripts/start_pd_v2.py --serial > /tmp/pd_test/_final.log 2>&1 < /dev/null &
# 3) 健康检查（图编译 ~3-8 分钟）
curl -s http://127.0.0.1:8000/health   # prefill (cards 0-7)
curl -s http://127.0.0.1:8001/health   # decode  (cards 8-15)
```

关键配置（start_pd_v2.py）：AscendStoreConnector + shm 后端 + lookup_rpc_port 14579；`enable_npugraph_ex:true, enable_static_kernel:false`（图模式+动态内核——当前唯一可用组合）；decode 侧 MTP=1 推测解码；`--enable-prefix-caching` 由 ascend 插件默认开启（日志确认 enable_prefix_caching=True）。
**注意**：launcher 日志加 `-u`（无缓冲），否则日志 0 字节；两个站点包补丁在镜像重建后必须重新执行。

## 4. 验证方法

- 两阶段验证客户端：`perf_opt/scripts/pd_4k_two_phase.py`（可改 prompt 长度；producer 先发 → 5s 后 consumer 同 prompt）
- 流水线基准：`perf_opt/scripts/pd_pipeline_bench.py --num-requests 32 --consumer-concurrency 32 --lookahead 4 [--post-delay 3]`
- 判别指标：consumer 日志 10s 窗口的 `Avg prompt throughput`（>0 = 本地重算；=0 = 池加载）；`/tmp/lookup_rpc_port_14579_dp_rank0` socket 存在 = producer 的 LookupKeyServer 已绑定

## 5. KV 池未打通的三个已定位原因（按修复优先级）

1. **16384 token 粒度闸门**（硬挡 <16K prompt）：`KVPoolScheduler.get_num_new_matched_tokens` 开头 `if token_len < self.cache_transfer_granularity: return 0, False`。granularity = lcm(128, 各组 128×compress_ratio)，DSV4 MLA 压缩组 ratio=128 → **16384**。`discard_partial_chunks=True` 时更会把 <16384 的 prompt 向下取整到 0。→ 实测 2772/4066 token 全部被挡；17524 token 能过闸门。
2. **producer 保存路径未触发**（≥16K 时 hit=0 的直接原因）：`AscendStoreConnector.save_kv_layer` 开头 `if not self.use_layerwise: return`；非 layerwise 模式下 model_runner 也不调用任何保存。→ **下一步：两侧 `kv_connector_extra_config` 加 `"use_layerwise": true` 再测**（17.5K prompt 的 KV ≈250MB，正好压在 251MB 池容量线上，可能需要同时调大 kv_buffer_size）。
3. 历史问题（已修）：full-hash 补丁（512 token 只命中首段）、consumer 自建 LookupKeyServer 冲突（pd_fix_connector.py）、静态内核误删（第 2 节）。

## 6. 实测数据记录（2026-10-28，动态内核 + 图模式）

| 实验 | 配置 | 结果 |
|:-----|:-----|:-----|
| 4K 两阶段（单请求） | 2768-token prompt，producer 64 tok / consumer 1024 tok | producer 3.2s；consumer TTFT 0.316s（=本地 prefill，非池命中） |
| 流水线 32×4066 | lookahead 4，500MB 池 | consumer TTFT p50 7.4s / p99 23.2s；输出 176.2 tok/s；零池命中（全量本地重算） |
| 流水线 32×4066 | lookahead 1 | TTFT p50 7.4s，无改善 |
| 流水线 32×4066 | 池 8GB | TTFT p50 9.5s（更差）；/dev/shm psm 实测仍 ~251MB（配置字段不控制实际分配） |
| 流水线 32×4066 | post-delay 3s | TTFT p50 8.4s，无改善 |
| 17.5K 两阶段 | 过 16384 闸门 | lookup 执行（2190 哈希）→ hit=0（保存未发生） |

诚实结论：**当前配置下 PD 无 TTFT 收益**；机制链路（调度→闸门→lookup→RPC→池）已全部摸清，剩余两个修复点明确（use_layerwise:true + 池容量）。

## 7. 收尾注意事项

- 工作完成后：站点包探针清除（如插过桩）、git 提交、**保存镜像**、**关闭实例并验证 STOPPED**（inspire notebook stop 后必须 status 复核）
- 实例重启不重置站点包与 /tmp；镜像重建才重置（重建后需重新打两个补丁）

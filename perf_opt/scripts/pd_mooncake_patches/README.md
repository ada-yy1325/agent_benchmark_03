# Mooncake P/D KV 传输完整打通——补丁集与应用顺序（2026-11-02 定稿）

**状态：KV 传输端到端打通、输出正确（consumer prompt throughput=0.0 铁证）。**
性能待优化（c16 TTFT p50 9.6s vs 基线 0.88s@c8——拉取排队+无 MTP，详见 logbook）。

## 环境依赖（10-30 已就位，勿动）

- `/opt/mamba/lib/python3.13/site-packages/mooncake/`：Cline 自编译 Ascend 版 engine.so + ascend_transport.so + 22 个依赖库副本（bak_libs/ 里是重复的默认路径库）
- 连接器原始文件 = vllm 0.23.0 官方版（jsdelivr 获取：`https://cdn.jsdelivr.net/gh/vllm-project/vllm@v0.23.0/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`）
- 引擎源码与构建树：`/inspire/sj-ssd3/project/project-public/s26068/mooncake-src/`（数据盘，持久）

## 补丁应用顺序（全部幂等设计）

1. **patch_engine.py** —— 引擎源码两处放宽重叠拒绝（registerLocalMemoryBatch 的排序检查 + tryReserveMemoryRegions），然后：
   `cd mooncake-src/build && cmake --build . -j 8 --target transfer_engine`
   把新 libtransfer_engine.so 复制进 mooncake 目录。
2. **patch_final_conn.py** —— 连接器：预加载 4 库（yaml-cpp+mooncake 三件套，绝对路径 ctypes 预加载、零 LD 污染）+ hybrid 列表展平 + **存储级注册**（引擎/ADXL 注册走 untyped_storage 去重、不重叠；逐组寻址列表 kv_caches_base_addr/block_len_per_layer 保持原样供传输公式使用）。
3. **patch_datalen.py** —— 传输拷贝长度 = 每组真实块数据大小（nbytes//shape[0]），步长保留用于寻址；kv_block_lens 字段贯穿元数据与 P/D 校验。
4. **patch_proto.py** —— launcher 配置 `"mooncake_protocol": "ascend"`（AscendDirect 设备传输；"rdma" 会跳过设备传输导致 TCP 读不了 NPU 内存）。
5. **patch_roce.py + patch_bufpool.py** —— launcher 环境 `HCCL_INTRA_ROCE_ENABLE=1`、`ASCEND_BUFFER_POOL=4:8`（容器无 RDMA 网卡时的官方备选）。
6. **patch_mtp.py** —— launcher 移除 decode 侧 MTP（P/D 区域数必须对称 64=64）。

## 客户端编排契约（官方协议，必须遵守）

- P 请求：`kv_transfer_params={"do_remote_decode": True, "transfer_id": <uuid>}`，**max_tokens=1**（违反会导致生成块被当 prompt 块传输→乱码）
- D 请求：`kv_transfer_params={"do_remote_prefill": True, "remote_engine_id": <从 P 的 http://127.0.0.1:8998/query 运行时获取>, "remote_bootstrap_addr": "http://127.0.0.1:8998", "transfer_id": <同 uuid>}`
- 参考客户端：远端 /tmp/pd_orch.py（单请求）与 /tmp/pd_orch_bench.py（并发压测）

## 关键实测数据（11-02）

- 单请求：P 0.26s（2768 token 全 prefill）→ D 2.2-8.7s（拉 KV + 生成），D prompt throughput=0.0，输出正确
- c16 并发：TTFT p50 9588ms / p99 26501ms，16/16 成功（吞吐计数 bug 已修待重测）
- 两侧区域数 64=64（PDCNT 日志验证）

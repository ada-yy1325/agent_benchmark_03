# PD 分离问题详细报告（2026-10-28，给其他 Agent）

## 目标

在昇腾 910B2C ×16 上给 DeepSeek-V4-Flash（W8A8 量化）做 PD（prefill/decode）分离，改善 TTFT 与 output tok/s。对比基准：双服务 MTP=ON 聚合 1588.6 tok/s @c32；统一服务器 c32 TTFT 4114ms。

## 环境

- 实例：quant-test（昇腾卡公共空间，16×910B2C 每卡 64GB，镜像 glm53-w8a8-env:v2）
- 模型：`models/DeepSeek-V4-Flash-w8a8-npu16`（280GB，好模型已验证）
- vLLM 0.23.0 + vllm-ascend；Python 3.13；远端代码 `/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test`（git main，最新 3cd1d50）
- Mooncake：`mooncake-transfer-engine` 0.3.13.post1 已安装，`import mooncake` OK；vllm 工厂已注册 `MooncakeConnector` / `MooncakeStoreConnector`（注意：v0.23.0 里没有教程说的 `MooncakeLayerwiseConnector`）；proxy 脚本在 `mooncake/vllm_v1_proxy_server.py`

## 核心问题 A：平台 NPU 环境回归（10-27 下午起）

**现象**：任何带图编译的 vLLM 服务器启动时，worker 在 `profile_run` 阶段崩溃。错误家族（同一崩溃的不同表现）：

```
RuntimeError: fill_: aclnnInplaceFillScalar failed, error code is 561103
FillAiCore ADD_TO_LAUNCHER_LIST_AICORE failed
RuntimeError: dynamic_quant_run_aclnn ... AclNN_Parameter_Error(EZ1001)
EZ9999 AclNN_Inner_Err
```

**排除实验**（全部做过）：三个节点（412/416/420）全复现；纯标准配置（无 PD 无补丁）也崩；关 static_kernel/npugraph_ex 无效；实例重启清零无效。10-27 上午与 10-26 同镜像同脚本完全正常 → 判定平台驱动/固件更新导致，与代码无关。

**已验证的绕过**：`--enforce-eager`（纯 eager，完全跳过图编译）——普通服务器（`start_server_phase5.py --mtp 0 --enforce-eager`）启动成功且输出正常。

## 核心问题 B：KV 传输连接器在 eager 下也触发崩溃（当前卡点）

**单变量实验**（基准=上述验证过的纯 eager 配置）：

| 实验 | 结果 |
|:-----|:-----|
| 基准（纯 eager，无 kv 配置） | ✅ 启动成功，输出 "Paris" |
| 基准 + `--kv-transfer-config MooncakeConnector`（唯一变量） | ❌ 同样 FillKernel/aclnn 崩溃（36 处），profile_run 阶段 |
| 基准 + `--kv-transfer-config AscendStoreConnector`（唯一变量） | ❌ 同样崩溃（P5 实测确认） |

即：**只要加上任何 kv-transfer-config，eager 也救不了**。两个连接器单变量实验均已崩溃。

**根因分析（2026-10-28 定稿）**：平台 10-27 下午更新后，一批 NPU 设备算子坏了（561103 属 aclnn 算子内部异常类）。KV 连接器初始化时需在 NPU 上填充/注册 KV buffer，恰好调用坏掉的 fill 算子（aclnnInplaceFillScalar）→ 任何连接器必崩。**这不是 connector 参数可调的问题，调参无法解决。**

**问题给下一位**：
1. 为什么 kv 连接器的存在会让 profile_run 崩溃？连接器初始化在崩溃前执行了什么算子（内存分配、memset、量化反量化）？
2. 能否绕过：例如让连接器"延迟初始化"（profile_run 之后再初始化）、给 Mooncake 用不同 transport/更小 buffer、或打补丁跳过 profile_run 中与连接器相关的路径？
3. Mooncake 引擎是否支持纯 TCP 模式（本地回环）以避免 RDMA 初始化路径？当前 kv_connector_extra_config 是 `{"prefill":{"dp_size":1,"tp_size":8},"decode":{"dp_size":1,"tp_size":8}}`，未指定 transport。

## 问题 C（已解决，供参考）：vLLM 原生 PD 不可行

`--data-parallel-size 2` 单部署：`AssertionError: expected OutputCode, got _CompiledFxGraph / GraphModuleImpl`（vllm-ascend 编译管线不支持 DP>1）。

## 问题 D（已解决一半）：AscendStore 池长 prompt 只命中 512 token

根因：V1 逐步填充 block_hashes。补丁 `scripts/pd_full_hash_fix.py` 已写好（幂等、py_compile 验证、自动回滚），但因问题 B 未能实测。

## 复现命令（供下一位直接跑）

```bash
# 1. 基准（会成功）
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 setsid nohup \
python3 perf_opt/scripts/start_server_phase5.py --port 8000 --mtp 0 --enforce-eager \
> /tmp/base.log 2>&1 < /dev/null &

# 2. 加 Mooncake（会崩，问题 B 复现）
python3 perf_opt/scripts/start_server_phase5.py --port 8000 --mtp 0 --enforce-eager \
  --extra-vllm-args '--kv-transfer-config {"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_port":"36000","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":8},"decode":{"dp_size":1,"tp_size":8}}}'

# 观察：/tmp/*.log 中 profile_run 阶段的 aclnn 错误
```

## 最终执行计划（2026-10-28 定稿）

1. 记录驱动/固件版本留证（version.info + npu-smi），供平台工单对比 10-26 状态
2. 试一次 MC_FORCE_TCP=1 排除 RDMA 变量（预期仍崩，只做一次、记录结果）
3. 阶段 6 收尾：/tmp 产物归集 git、final report 固化 1588.6 tok/s 成果；tag perf-opt-final
4. PD 待平台确认修复后按 PD_HANDOFF_1028.md 第 6 节执行

## 平台恢复后的完整验证步骤

见 `perf_opt/PD_HANDOFF_1028.md` 第 6 节（补丁→串行启动→4K 两阶段→流水线基准→对比）。

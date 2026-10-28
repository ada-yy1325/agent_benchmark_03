# PD 分离问题详细报告（2026-10-28，给其他 Agent）

## 🎯 根因已定案（2026-10-28 下午，最终结论）

**所有 fill/aclnn 崩溃的真正原因：`start_pd_v2.py` 启动时的"清理"步骤误删了静态内核包。**

证据链（全部实测）：
1. `start_pd_v2.py` 每次启动执行 `rm -rf /usr/local/Ascend/cann-*/opp/static_kernel/static_kernel_*`——把 10-26 镜像构建时用 bisheng-compiler 编译的 **8 个静态内核包**（`static_kernel_261026*_*`）全部删除。
2. 该平台运行时**不会再生**这些内核；任何 aclnn 算子走静态路径时按 `config/ascend910b/binary_info_config.json` 查找 binPath → 文件不存在 → `561103` / `FillAiCore ADD_TO_LAUNCHER_LIST_AICORE failed` 硬崩（不回退动态编译）。
3. 独立最小复现：`torch.zeros(1).fill_()` 在配置完好时通过；图捕获（`torch_npu.npu.graph` 上下文中含 fill）在 binary_info_config.json 仍含 Fill 条目时必崩。
4. 与连接器（Mooncake/AscendStore）、eager/图模式、multistream、节点（412/416/420）**均无关**——之前"连接器必崩"是混杂变量误判（所有连接器实验恰好都走了带清理的启动路径或删内核后的环境；E1 成功是因为手动启动未清理）。
5. 修复（三处最小改动，已全部验证）：
   - `config.ini` 备份并清空 `load_priority`（原文件备份为 `config.ini.bak`）
   - `binary_info_config.json` 备份并清空为 `{}`（备份为 `.bak`）
   - `start_pd_v2.py` / `start_pd_v3.py` 的清理命令改为无害 `ls`（防再犯）
   - `start_pd_v2.py` 的 `enable_static_kernel: true → false`（动态 JIT 跑通；静态内核后续可用 torch_npu 的 compile_static_kernel 再生恢复性能）
6. 验证结果：迷你图捕获测试 `CAPTURE+FILL OK`；PD 双服务器 FINAL6 启动成功（decode 8001 + prefill 8000 双 200）；4K 两阶段验证 **consumer TTFT=0.316s**（producer 本地 prefill ~1.4s），2768-token prompt 全量命中池，**full-hash 补丁生效**。

> **更正（10-28 晚间探针实测）**：上一条"consumer TTFT 0.316s = 池命中"是误读——producer 自己的 TTFT 指标同为 0.308s，即两者都是本地 prefill（空闲单请求 ~9K tok/s，2768 token ≈ 0.3s）。池从未工作过。

## 🎯 KV 池未打通的三个原因（2026-10-28 晚间定稿，探针矩阵实测）

探针法（调度器调用点→函数入口→lookup 后→producer 应答者四处插桩）定位结果：

1. **16384 token 粒度闸门**：`cache_transfer_granularity = lcm(128, 各组 128×compress_ratio)`，DSV4 MLA 压缩组 ratio=128 → 16384。`get_num_new_matched_tokens` 对 `token_len < 16384` 直接 return 0（discard_partial_chunks 时先向下取整）。→ 我们测的 2772/4066 token 全部被挡，从未 lookup。
2. **producer 保存路径未触发**：`save_kv_layer` 开头 `if not self.use_layerwise: return`；非 layerwise 模式下无任何保存执行。17.5K prompt 过了闸门、lookup 真正执行（2190 哈希）→ hit=0，池空。
3. **kv_buffer_size 配置不生效**：/dev/shm psm 实测恒 ~251MB（8GB 配置亦然）；17.5K 请求的 KV ≈250MB 已贴容量线。

修复建议（给下一位）：① 两侧 `kv_connector_extra_config` 加 `"use_layerwise": true` 激活逐层保存；② 用 ≥16384 token prompt 验证；③ 池容量若可调再加大。

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

## 决定性对照实验矩阵（2026-10-28 完成，全部在 node 416 全新容器上）

| # | 配置（从冻结基准逐次加变量） | 结果 |
|:-:|:-----|:-----|
| 冻结 | 10-26 tag 代码 + 无补丁 + 图模式 + MTP=1 + static + fp8 KV + multistream ON | ✅ 成功（输出 "Paris"） |
| B1 | + pd_full_hash_fix 补丁 | ✅ 成功（补丁无罪） |
| B3 | Mooncake v2 启动器全套（eager） | ❌ worker 初始化阶段崩（AclNN_Inner_Err） |
| P4 | 冻结 + eager + MooncakeConnector | ❌（36 处 fill/aclnn） |
| P5 | 冻结 + eager + AscendStoreConnector | ❌ |
| B4 | 冻结 + eager + AscendStore + **multistream OFF** | ❌（54 处） |

**结论更新**：
1. 平台"坏算子"结论**修正**：常规推理路径（含图模式）在干净容器上完全正常——FAQ 的"图捕获过多"理论可解释部分图模式崩溃，但无法解释 eager+连接器崩溃。
2. **唯一稳定触发条件 = kv-transfer-config（连接器）存在**：无论 Mooncake/AscendStore、无论 eager/图模式、无论 multistream 开关，连接器初始化必崩。冻结配置则一切正常。
3. 下一步建议（给下一位）：定位连接器初始化路径中调用了哪些 aclnn 算子（worker init 阶段），对照常规路径的算子使用差异；或在连接器 init 代码中打日志二分具体崩溃算子。

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

## 最终执行计划（2026-10-28 定稿，已执行完毕）

1. ✅ 版本留证：`env/platform_evidence_1028.md`（npu-smi 25.0.rc1.1 / torch_npu 2.11.0.rc4 / CANN 9.0.0 / 节点 416）
2. ✅ MC_FORCE_TCP=1 单次实验（已记录，不重复）：失败方式不同——`Mooncake is not available`（NPU 版引擎 TCP 模式不可用），worker 侧仍伴随 fill 崩溃。排除 RDMA 变量无果，与根因判断一致
3. ✅ 阶段 6 收尾完成（git 5265819/977037d；1588.6 tok/s 成果已固化）
4. ⏳ PD 待平台确认修复后按 PD_HANDOFF_1028.md 第 6 节执行

## 平台恢复后的完整验证步骤

见 `perf_opt/PD_HANDOFF_1028.md` 第 6 节（补丁→串行启动→4K 两阶段→流水线基准→对比）。

# PD 分离任务交接书（2026-11-10，v3 最新版）

> **覆盖所有旧版本**。读完即可全面了解 PD 分离的**现状、已排除路径、已知问题、下一步**。

---

## 〇、当前状态（11-10 10:00）

- **CLI 连通性**：✅ 已修复。`~/.inspire/notebook-gpu-models.json` 中 `gpu_model` 设空改为 `"H100"`（workaround，强迫 JupyterTerminal 通道）。`inspire notebook exec` 恢复正常。
- **实例 quant-test**：RUNNING，16×910B2C 全部健康，无残留进程。
- **代码**：GitHub `main` 分支 commit `0c194a4`（最新），远端已同步。
- **P/D 服务**：当前未在运行。每次实例重启 tmux session 全丢，需手启。
- **上次成功状态**：11-09 晚最后一次测试，P 和 D 均稳定启动（不崩溃），bootstrap 8998 有响应，P connector 在收到 `do_remote_decode=true + transfer_id` 后 **正确进入 send path**（`_reqs_need_send` 入队成功）。但端到端 KV 传输 **尚未打通**。

---

## 一、从 11-06 到 11-10 的所有工作

### 1.1 配置修正（5 项，已 git 推送）

| # | 修正 | 状态 | 原因 |
|:--|:--|:--|:--|
| 1 | 删除 `--enforce-eager` | ✅ | 与 cudagraph 冲突，cudagraph 实测快 ~300ms |
| 2 | 删除 P 侧 `--speculative-config` (MTP) | ✅ | P 只出 1 token，MTP 无意义 |
| 3 | `kv_connector`: 先改 `MooncakeConnectorV1` → 回退 `MooncakeConnector` | ✅ | vllm-ascend 版缺 11-02 补丁；vllm base 版有补丁 |
| 4 | 删除 `kv_ip: "127.0.0.1"` | ✅ | 官方指南无此字段 |
| 5 | 新增 `mooncake_protocol` 字段 | ✅ | 见下节 |

### 1.2 协议之争：ascend vs tcp（已彻底验证）

| 路径 | MC_FORCE_TCP | protocol | 结果 | 根因 |
|:--|:--|:--|:--|:--|
| **Ascend Direct** | 不设 | `ascend` | ❌ 全部 8 个 worker 报 103900 | `HcclRegisterGlobalMem` 拒绝 torch_npu KV buffer——**包括 TP0**。PyTorch allocator 内存与 ADXL/HCCL 不兼容。不是"兄弟进程重叠"问题。 |
| **TCP + 完整补丁** | `1` | `tcp` | ✅ P/D 启动不崩溃，`Registering KV_Caches` 正常，`batch_register_memory` 无报错 | `posix_memalign` 配 connector 补丁可安全处理 NPU 内存 |
| **TCP + 无补丁** | `1` | — | N/A（未单独测） | — |

**当前配置**：`MC_FORCE_TCP=1` + `mooncake_protocol=tcp`

### 1.3 Connector 补丁状态

**关键事实**：
- 容器重建前 connector 补丁在镜像层，重建后需每次重打。
- `patch_final_conn.py` 的 anchor 依赖 `/tmp/mc_0230.py`（pristine v0.23.0 原版）。此文件在 /tmp，但截至 11-10 还未丢。
- `patch_proto.py` 和 `patch_mtp.py` 的 **anchor 已失效**（11-06 脚本改了 `start_pd_mooncake.py` 的结构），不能机械套用。
- `patch_datalen.py` 的 anchor 在补丁后的 connector 中变了，但第一轮（先打 final_conn、后打 datalen）时仍匹配。
- **打补丁顺序必须**：`patch_final_conn.py` → `patch_datalen.py` → `pd_add_traces.py`（先 final_conn 恢复原版，再打其他补丁，否则会被覆盖）。

**补丁核对表**（每次检查）：
- `grep -c "PD fix" /opt/mamba/lib/python3.13/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py` → 应输出 `1`
- `grep -c "PDDBG" <同上>` → 应输出 `1`（插桩存在）
- `/tmp/mc_0230.py` 应存在（文件大小 72414 字节）
- `find /opt/mamba/lib/python3.13/site-packages/mooncake/ -maxdepth 1 -name '*.so' | wc -l` → 应输出 `49`
- `libtransfer_engine.so` 应为 Nov 2 11:01（11-02 修复版）

### 1.4 Proxy 改造（已 git 推送 v2）

**原 Proxy 的错误假设**：以为 P 返回 `kv_transfer_params`，再转发给 D。

**11-09 发现**：P 在 HTTP 响应中**永远**返回 `kv_transfer_params: null`——这是正确行为。vLLM connector 在 EngineCore 内部 push KV，不经过 HTTP 响应。

**v2 Proxy（`pd_proxy.py`）三阶段**：
1. Stage 1: 发 `do_remote_decode=true + transfer_id` 到 P（max_tokens=1）
2. Stage 2: 请求 `http://127.0.0.1:8998/query` 取 P 的 `engine_id`
3. Stage 3: 发 `do_remote_prefill=true + engine_id + bootstrap_addr` 到 D

**此 v2 proxy 尚未在远端测试过**。但单独 curl 直连 P 验证了 Stage 1 的信息流是对的。

### 1.5 11-09 插桩验证的关键发现（最重要）

**直连 P 打 curl**：
```json
{"do_remote_decode":true, "do_remote_prefill":false, "transfer_id":"xfer-test-xxx"}
```

**P connector 日志**（INFO 级别，确认 params 完整传入了 EngineCore）：
```
PDDBG update_state_after_alloc req=cmpl-xxx kv_transfer_params={'do_remote_decode': True, 'do_remote_prefill': False, 'transfer_id': 'xfer-test-xxx'}
PD-DEBUG do_remote_decode branch entered, transfer_id=xfer-test-xxx
PD-DEBUG _reqs_need_send queued req_id=cmpl-xxx
```

**结论**：P connector 的信息流从 API → EngineCore → Scheduler → Connector 全程正确。
**params 不是 None，字段正确送达。**

---

## 二、当前卡点

### 卡点 1：Stage 2→3 从没用新 proxy 完整跑过

v2 proxy 只是肉眼检查了代码，**远端实际尚未启动 proxy**。需要：

1. 拉代码
2. 打所有补丁（final_conn → datalen → traces）
3. 重生成启动脚本（确认 MC_FORCE_TCP=1 + protocol=tcp）
4. 先起 D、等健康、起 P、等健康、确认 8998 有响应
5. 起 proxy v2
6. 通过 proxy 发请求
7. 检查 D metrics `external_kv_transfer > 0` + `Avg prompt throughput: 0.0`

### 卡点 2：MTP 两侧对称（未验证）

11-02 logbook 提到 MTP 两侧必须对称（P 和 D 都 64 blocks）。之前 Cline 单独删了 P 的 `--speculative-config`，但没确认 D 还在不在。

原本计划进一步修正 `start_pd_mooncake.py` 给 D 单独加 MTP（只对 decode 有用），但当前阶段目标是 **打通 KV 传输**，MTP 是性能问题可以延后处理。

---

## 三、已排除的路径

1. ❌ Ascend Direct (`protocol="ascend"`, 无 MC_FORCE_TCP)：`HcclRegisterGlobalMem err=103900`，全部 8 个 worker（包括 TP0）都失败
2. ❌ 修改 `ascend_direct_transport.cpp` 容忍 103900：修改后 via cmake rebuild，但仍失败——错误来自 HCCL 底层，不在 transport 层
3. ❌ 全局 DEBUG 模式：`VLLM_LOGGING_LEVEL=DEBUG` 触发 `npugraph_ex._CompiledFxGraph != OutputCode` 断言崩溃（非 PD 问题，是 ascend 已知限制）
4. ❌ `patch_proto.py`：anchor 已失效，且协议已正确设为 tcp
5. ❌ `patch_mtp.py`：anchor 已失效
6. ❌ 读 P 的 HTTP 响应里的 `kv_transfer_params`：不可能有值，这是正常行为
7. ❌ Proxy 不带 `transfer_id`：已修，报 `Missing transfer_id` 表明 do_remote_decode 确实被检测到了

---

## 四、下一步（助手可直执行，不需确认）

### 4.1 最小部署 + 测试序列

```bash
# 0) 清理 + 拉代码
cd /inspire/.../agent_benchmark_test && git fetch origin && git reset --hard origin/main

# 1) 打补丁 + 插桩
python3 perf_opt/scripts/pd_mooncake_patches/patch_final_conn.py
python3 perf_opt/scripts/pd_mooncake_patches/patch_datalen.py
python3 perf_opt/scripts/pd_add_traces.py

# 2) 重生成启动脚本（确认 tcp + MC_FORCE_TCP=1）
python3 -c "
import json,os,sys;sys.path.insert(0,'perf_opt/scripts')
from start_pd_mooncake import build_args,generate_shell_script
TMP='/tmp/pd_test';os.makedirs(TMP,exist_ok=True)
for pf,port,lb in [(False,8001,'decode'),(True,8000,'prefill')]:
    a=build_args(prefill=pf,port=port)
    json.dump(a,open(TMP+'/pd_'+lb+'_args.json','w'))
for role,devs,log,af in [
    ('pd_decode','8,9,10,11,12,13,14,15','decode_v2.log','pd_decode_args.json'),
    ('pd_prefill','0,1,2,3,4,5,6,7','prefill_v2.log','pd_prefill_args.json')]:
    s=generate_shell_script(role,devs,TMP+'/'+af,TMP+'/'+log)
    with open(TMP+'/_'+role+'_launch.sh','w') as f:f.write(s)
    os.chmod(TMP+'/_'+role+'_launch.sh',0o755)
print('OK')
"

# 3) 起 D → 等 200 → 起 P → 等 200 → 确认 8998/query
tmux new-session -d -s pd_decode bash && sleep 0.5 && \
  tmux send-keys -t pd_decode 'bash /tmp/pd_test/_pd_decode_launch.sh' Enter
# 等待 D 健康 (while curl | D 健康检查 | sleep 10)
tmux new-session -d -s pd_prefill bash && sleep 0.5 && \
  tmux send-keys -t pd_prefill 'bash /tmp/pd_test/_pd_prefill_launch.sh' Enter
# 等待 P 健康
curl -s http://127.0.0.1:8998/query  # 确认 bootstrap 活着

# 4) 起 proxy v2
nohup python3 perf_opt/scripts/pd_proxy.py --prefill-url http://127.0.0.1:8000 \
  --decode-url http://127.0.0.1:8001 --port 8002 > /tmp/pd_proxy.log 2>&1 &

# 5) 成对测试
curl -s -X POST http://127.0.0.1:8002/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"dsv4","prompt":"The capital of France is","max_tokens":20}'

# 6) 四条铁证
curl -s http://127.0.0.1:8001/metrics | grep 'prompt_tokens_by_source'
grep 'Avg prompt throughput' /tmp/pd_test/decode_v2.log | tail -2
```

### 4.2 关键检查点

| # | 检查项 | 预期 |
|:--|:--|:--|
| 1 | `grep "MC_FORCE_TCP" /tmp/pd_test/_pd_prefill_launch.sh` | 输出 1（存在） |
| 2 | `grep "mooncake_protocol" /tmp/pd_test/pd_prefill_args.json` | `tcp` |
| 3 | P 日志 `PDDBG update_state_after_alloc` 有输出 | params 非 None |
| 4 | P 日志 `_reqs_need_send queued` 有输出 | send path 入队 |
| 5 | D metrics `external_kv_transfer > 0` | KV 被远程拉取 |
| 6 | D 日志 `Avg prompt throughput: 0.0 tokens/s` | D 未本地 prefill（唯一铁证） |

---

## 五、环境清单（每次启动前检查）

| 项 | 检查命令 | 预期 |
|:--|:--|:--|
| CLI 连通 | `inspire notebook exec quant-test ... "echo ok"` | ok |
| 实例状态 | `inspire notebook list` | RUNNING |
| NPU 空闲 | `npu-smi info \| grep 910B2C \| wc -l` | 16 |
| 端口空闲 | `ss -tlnp \| grep -E '800[0-2]'` | 无输出 |
| 代码最新 | `git log --oneline -1` | 最新 commit |
| connector 补丁 | `grep -c "PD fix" /opt/.../mooncake_connector.py` | 1 |
| pristine 原版 | `ls -la /tmp/mc_0230.py` | 72414 bytes |
| .so 数量 | `find /opt/.../mooncake/ -maxdepth 1 -name '*.so' \| wc -l` | 49 |
| libtransfer_engine | `ls -la /opt/.../mooncake/libtransfer_engine.so` | Nov 2 11:01 |
| bak_libs | `ls /opt/.../mooncake/bak_libs/` | 3 个文件（libascend_hal.so, libc_sec.so, libmmpa.so） |

---

## 六、注意事项（给下一位）

1. **先读环境/补丁状态再动配置**：`/tmp` 在容器重启不丢，但所有 tmux 进程全丢。启动前必须跑「环境清单」。
2. **打补丁顺序**：`patch_final_conn.py`（每次都从 /tmp/mc_0230.py 恢复原版）→ `patch_datalen.py` → `pd_add_traces.py`。
3. **MTP**：当前 D 不带 MTP。如果后期要加，记得 P 不要加（MTP 对 prefill 无意义），D 可以用 `--speculative-config`。
4. **不要用全局 DEBUG**：`VLLM_LOGGING_LEVEL=DEBUG` 触发 `npugraph_ex` 编译断言崩溃。
5. **不要尝试 Ascend Direct**：HcclRegisterGlobalMem 拒绝 torch_npu KV buffer（103900），TCP 模式是唯一可用路径。
6. **proxy v2 还未经实战验证**：Stage 2 (8998/query) 和 Stage 3 (do_remote_prefill → D) 的接合逻辑肉眼检查过但未曾实测。
7. **`/tmp/pd_test/` 下有过往诊断日志**：prefill_v2.log、decode_v2.log 是历史（11-09），可作参考。
8. **平台有命令长度限制**：`inspire notebook exec` 传过去的单条命令太长会被截断，不确定具体上限。建议分步执行，每步短小。

---

## 七、长期目标

PD 分离在当前 16 卡单节点对 DSV4 W8A8 的优化收益为负（详见 logbook 11-03 跨场景全表），本次任务是**验证 KV 传输打通**、不是做性能优化。打通后应记录证据、固化环境、收尾关实例。

---

*最后更新：2026-11-10 | 状态：等了 10+ 小时重启后环境完好，代码/补丁就位，proxy v2 待首次测试*
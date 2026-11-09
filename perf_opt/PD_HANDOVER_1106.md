# PD 分离任务交接书（2026-11-09，最终版）

> 此文档覆盖之前所有版本。读完此文档即可全面了解 PD 分离任务的**现状、问题、进度、下一步**。

---

## 〇、当前阻塞（一句话）

**inspire CLI 连接远程实例 `quant-test` 失败**——实例已 RUNNING（16×910B, GPU 0%），但 `inspire notebook exec` 卡死无响应，连 `echo hello` 都不返回。怀疑是 11-09 实例重新启动后 CLI 缓存的 SSH 通道信息过期。此问题解决后，以下配置修正 + 启动 + 验证可一小时内完成。

---

## 一、PD 分离进度总结

### 1.1 已跑通的

| 项目 | 状态 | 详情 |
|:--|:--|:--|
| P/D 双服务启动 | ✅ 之前跑通过 | 通过 `start_pd_mooncake.py --serial` (tmux) |
| c1/c4/c8/c16 并发扫射 | ✅ 11-04 完成 | 吞吐 39.5→149.8→274.6→486.2 tok/s |
| KV 传输参数配置 | ⚠️ 配了但**没生效** | `external_kv_transfer = 0` |
| Proxy 代码 | ✅ 已写好 | `pd_proxy.py` (152 行) |
| 官方参考资料 | ✅ 已下载 | `mooncake_guide_official.md` + `pd_proxy_official.py` |

### 1.2 致命发现（11-06）

**PD 分离一直没真正工作！** 证据来自 D 的 `/metrics` 端点：

```
prompt_tokens_by_source{source="external_kv_transfer"} = 0.0   ← D 从未拉取 P 的 KV
Num KV expired reqs = 32                                       ← P 产的 KV 块 480s 后过期
```

**根因**：之前没有 Proxy。用户直接请求 D，D 只用本地 prefill，KV 传输形同虚设。

### 1.3 本次（11-09）新发现的配置问题

完整阅读 `start_pd_mooncake.py`（362 行）和官方指南后，发现 5 个待修正之处：

#### ❌ 问题 1：`--enforce-eager` 与 cudagraph 冲突

```python
# Line 182: 禁用了 cudagraph
"--enforce-eager",
# Line 185: 又开了 cudagraph（被上面覆盖，无效！）
"--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
```

**修正**：去掉 `--enforce-eager`，保留 cudagraph（11-03 测试证实 cudagraph 比 eager 快 ~300ms）。

#### ❌ 问题 2：P 侧加了 MTP speculative decode

P 的 build_args 中有 `if prefill:` 分支加了 `--speculative-config`，Prefill 只出 1 token，MTP 对它完全无意义。

**修正**：删除 P 的 `--speculative-config`。D 侧如需 MTP 单独加。

#### ⚠️ 问题 3：`kv_connector` 名称需确认

当前用 `"MooncakeConnector"`，官方 v0.10.1 指南用 `"MooncakeConnectorV1"`。需远程确认 v0.23.0 正确名称。

#### ⚠️ 问题 4：kv_config 多余/缺失字段

删 `"kv_ip": "127.0.0.1"`（官方无此字段），加 `"engine_id": rank`（官方有）。

#### ⚠️ 问题 5：DP 参数在 dp=1 时多余

`--data-parallel-address` / `--data-parallel-rpc-port` 在 dp=1 时无意义，保留无害但可清理。

---

## 二、当前架构（修正后）

```
                        ┌─────────────────┐
                        │   pd_proxy.py   │
                        │   :8002          │
                        └───┬─────────┬───┘
                   Stage 1  │         │  Stage 2
                   (prefill)│         │  (decode + kv_transfer_params)
              ┌─────────────▼──┐  ┌──▼─────────────┐
              │  P (prefill)   │  │  D (decode)    │
              │  Port: 8000    │  │  Port: 8001    │
              │  NPU: 0-7      │  │  NPU: 8-15     │
              │  tp=8          │  │  tp=8          │
              │  kv_producer   │  │  kv_consumer   │
              │  kv_port=20001 │  │  kv_port=20002 │
              │  engine_id=0   │  │  engine_id=1   │
              └────────────────┘  └────────────────┘
```

**数据流**：
1. Client → Proxy(:8002) `/v1/completions`
2. Proxy → P(:8000) prefill（P 做前向传播，存 KV 到 NPU buffer）
3. P 返回 response（含 `kv_transfer_params`）
4. Proxy 取出 `kv_transfer_params` → 注入请求 → D(:8001)
5. D 通过 Mooncake 传输层从 P 拉 KV 块 → 做 decode（生成 tokens）
6. Proxy 透传流式响应到 Client

**成功验证标志**：
```bash
curl -s http://127.0.0.1:8001/metrics | grep 'external_kv_transfer'
# 期望：prompt_tokens_by_source_total{source="external_kv_transfer"} > 0
```

---

## 三、远程连通性修复（当前阻塞）

### 现象
- `inspire notebook list` → quant-test 显示 **RUNNING**
- `inspire notebook metrics` → GPU 0%，无残留进程
- `inspire notebook exec` → **卡死**，连 `echo hello` 都不返回

### 已尝试
- 升级 inspire CLI：v7.1.14 → v7.1.16 ✅
- 杀掉所有卡死进程（pkill -9 inspire）✅
- 多次重试 exec → 全部失败

### 下一步尝试（按优先级）
1. **`inspire notebook stop` + `inspire notebook start`** 完全重启实例
2. **`inspire notebook shell`** 交互终端（不同的连接方式）
3. **清除 inspire 缓存**：`rm -rf ~/.inspire/cache*`
4. 如果以上都无效，**通过启智 Web 控制台的 Web Terminal** 直接进入容器

---

## 四、README：连接恢复后正确流程

### 4.1 启动全栈（严格顺序）

```bash
# Step 0: Git pull 最新代码
cd /inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test
git fetch origin && git reset --hard origin/main

# Step 1: 启动 D（decode）
tmux new-session -d -s pd_d \
  "ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
   python3 perf_opt/scripts/start_pd_mooncake.py 2>&1 | tee /tmp/pd_d.log"

while ! curl -s http://127.0.0.1:8001/health > /dev/null 2>&1; do
    echo "$(date): waiting for D..."; sleep 10
done
echo "✅ D healthy (8001)"

# Step 2: 启动 P（prefill）
tmux new-session -d -s pd_p \
  "ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
   python3 perf_opt/scripts/start_pd_mooncake.py 2>&1 | tee /tmp/pd_p.log"

while ! curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; do
    echo "$(date): waiting for P..."; sleep 10
done
echo "✅ P healthy (8000)"

# Step 3: 启动 Proxy
nohup python3 perf_opt/scripts/pd_proxy.py \
  --prefill-url http://127.0.0.1:8000 \
  --decode-url http://127.0.0.1:8001 \
  --port 8002 > /tmp/pd_proxy.log 2>&1 &
echo "✅ Proxy started (8002)"

# Step 4: 验证 KV 传输
curl -s -X POST http://127.0.0.1:8002/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"dsv4","prompt":"Hello, how are you?","max_tokens":20}'

curl -s http://127.0.0.1:8001/metrics | grep -E "external_kv_transfer|kv_expired"
# 期望: external_kv_transfer > 0
```

### 4.2 跑并发基准

```bash
for c in 1 4 8 16 32; do
    python3 perf_opt/scripts/pd_conc_bench.py \
      --prefill-url http://127.0.0.1:8000 \
      --decode-url http://127.0.0.1:8001 \
      --p-concurrency $c --d-concurrency $c --num-requests 64 \
      --input-chars 4000 2>&1 | tee /tmp/pd_bench_c${c}.log
done
```

### 4.3 关闭

```bash
tmux kill-session -t pd_p 2>/dev/null
tmux kill-session -t pd_d 2>/dev/null
pkill -f pd_proxy.py

# 保存结果
mkdir -p /inspire/.../perf_opt/results/pd_1109/
cp /tmp/pd_bench_c*.log /inspire/.../perf_opt/results/pd_1109/
```
---

## 五、待修正的代码（start_pd_mooncake.py）

### 5.1 build_args() 改动

删除 `--enforce-eager`（line 182）和 P 的 `--speculative-config`（lines 193-202）。
保留 cudagraph（`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`）。

### 5.2 build_kv_config() 改动

删除 `"kv_ip": "127.0.0.1"`（line 155），增加 `"engine_id": rank`。
确认 `kv_connector` 名称（`MooncakeConnector` vs `MooncakeConnectorV1`）。

### 5.3 改动对照

| 行号 | 旧值 | 新值 | 原因 |
|:--|:--|:--|:--|
| 182 | `"--enforce-eager",` | *(删除)* | 与 cudagraph 冲突；cudagraph 更快 |
| 193-202 | P 的 MTP speculative config | *(删除)* | P 只出 1 token，MTP 无意义 |
| 151 | `"MooncakeConnector"` | 需远程确认 | 可能应为 `MooncakeConnectorV1` |
| 155 | `"kv_ip": "127.0.0.1"` | *(删除)* | 官方指南无此字段 |
| — | *(缺失)* | `"engine_id": rank` | 官方指南有；P=0, D=1 |

---

## 六、参考资料

| 文件 | 路径 | 用途 |
|:--|:--|:--|
| 官方 Mooncake 指南 | `perf_opt/scripts/mooncake_guide_official.md` | v0.10.1 官方部署指南 |
| 官方 Proxy 参考 | `perf_opt/scripts/pd_proxy_official.py` | 1214 行完整实现 |
| 当前 PD 启动器 | `perf_opt/scripts/start_pd_mooncake.py` | 362 行，需改 3 处 |
| 最小 Proxy | `perf_opt/scripts/pd_proxy.py` | 152 行，1P1D 代理 |
| 后台测试脚本 | `perf_opt/scripts/pd_test_background.py` | 自动化验证 |
| 并发基准 | `perf_opt/scripts/pd_conc_bench.py` | c1-c32 扫射 |
| Logbook | `logbook.md` | 最新条目 11-06（第十回合） |

---

## 七、快速状态检查清单

恢复连接后，跑以下命令确认环境正常：

```bash
# 1. NPU 可用
npu-smi info | head -10

# 2. Python 正常
python3 --version && python3 -c "import vllm; print(vllm.__version__)"

# 3. 模型存在
ls /inspire/.../models/DeepSeek-V4-Flash-w8a8-npu16/*.safetensors | head -3

# 4. 代码最新
cd /inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test
git log --oneline -3

# 5. 无残留进程
ps aux | grep -E "vllm|VLLM" | grep -v grep
ss -tlnp | grep -E "800[0-2]"

# 6. Mooncake lib 完整
ls /opt/mamba/lib/python3.13/site-packages/mooncake/engine.so 2>/dev/null \
  && echo "engine.so OK" || echo "engine.so MISSING"
```

---

*最后更新：2026-11-09 | 作者：Cline Agent | 状态：等待远程恢复后执行第四章*

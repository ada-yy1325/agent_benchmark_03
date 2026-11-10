# P/D Separation Deployment Package

DeepSeek-V4-Flash-w8a8 推理服务 **Prefill-Decode 分离部署**，基于 vLLM + Mooncake KV Transfer，16 × Ascend 910B2C NPU。

> **两套可用方案，切换只需一条命令。**

---

## 快速参考

| 场景 | 启动命令 | 验证命令 | 压测命令 |
|:--|:--|:--|:--|
| **P/D No-MTP** | `bash pd_deployment/non_mtp/launch.sh` | `python3 pd_deployment/scripts/verify_pd.py` | `python3 pd_deployment/scripts/run_benchmark.py --no-mtp` |
| **P/D MTP ON** | `bash pd_deployment/mtp/launch.sh` | `python3 pd_deployment/scripts/verify_pd.py --mtp` | `python3 pd_deployment/scripts/run_benchmark.py --mtp` |

---

## 目录结构

```
pd_deployment/
├── README.md                    ← 本文件（使用手册）
├── SUMMARY.md                   ← 两版本对比数据 & 结论
│
├── non_mtp/                     ← 版本 1：基础 P/D 分离
│   ├── launch.sh                │   一键启动
│   ├── pd_prefill_args.ref.json │   P 端已知工作参数
│   └── pd_decode_args.ref.json  │   D 端已知工作参数
│
├── mtp/                         ← 版本 2：P/D + MTP 推测解码
│   ├── launch.sh                │   一键启动
│   ├── pd_prefill_mtp_args.ref.json │ P+MTP 端已知工作参数
│   └── pd_decode_mtp_args.ref.json  │ D+MTP 端已知工作参数
│
├── scripts/                     ← 所有工具脚本
│   ├── verify_pd.py             │   验证 P/D 是否真正分离
│   ├── run_benchmark.py         │   跑 benchmark 矩阵
│   ├── start_pd_mooncake.py     │   主启动脚本（被 launch.sh 调用）
│   ├── regenerate_pd.py         │   重生成 No-MTP args
│   ├── regenerate_pd_mtp.py     │   重生成 MTP args
│   ├── bench_serving.py         │   HTTP benchmark 客户端
│   └── pd_read_results.py       │   读取所有结果并打印汇总表
│
└── results/                     ← 压测结果存档目录
    ├── non_mtp/                 ← No-MTP 版本结果
    └── mtp/                     ← MTP 版本结果
```

---

## 前置条件

### 硬件 & 环境
- **16 × Ascend 910B2C NPU**（每卡 64 GB HBM）
- **CANN 9.0.0 + PyTorch 2.11 + vLLM 0.23.0 + vllm-ascend**
- **Mooncake engine v0.1**（`/opt/mamba/lib/python3.13/site-packages/mooncake/`）
- 镜像：`glm53-w8a8-env:v2`

### 模型
- 路径：`/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-npu16`（280 GB，w8a8 量化）

### Git
- 仓库：`https://github.com/ada-yy1325/agent_benchmark_03.git`
---

## 使用步骤

### Step 1: 启动 P/D

```bash
# 方式 A: No-MTP（基础分离，8 卡 P + 8 卡 D）
bash pd_deployment/non_mtp/launch.sh

# 方式 B: MTP ON（分离 + 推测解码，推荐）
bash pd_deployment/mtp/launch.sh
```

launch.sh 自动执行：git pull → 杀旧进程 → regenerate args → 启动 D (NPU 8-15, :8001) → 启动 P (NPU 0-7, :8000) → 等待双健康。
**预计耗时**：首次 ~5-8min（模型加载+编译），后续 ~3min（有缓存）。

### Step 2: 验证 P/D 真正分离

```bash
python3 pd_deployment/scripts/verify_pd.py        # No-MTP
python3 pd_deployment/scripts/verify_pd.py --mtp  # MTP
```

验证三阶段：
1. ✅ 两个实例 `/health` 返回 200
2. ✅ 发送测试请求返回正常输出
3. ✅ Decode 日志 `prompt throughput ≈ 0` → **PD 分离铁证**

### Step 3: 跑 Benchmark

```bash
python3 pd_deployment/scripts/run_benchmark.py --no-mtp --quick   # 冒烟 ~2min
python3 pd_deployment/scripts/run_benchmark.py --mtp               # 完整矩阵
```

矩阵覆盖：4K→128、16K→128、4K→1K，每个场景 c=1/4/8/16，16 请求/点。

### Step 4: 查看结果

```bash
python3 pd_deployment/scripts/pd_read_results.py
```

---

## 架构

```
用户请求 → Prefill (NPU 0-7, :8000, kv_producer)
                │  Mooncake TCP loopback KV transfer
                ▼
           Decode (NPU 8-15, :8001, kv_consumer) → 输出 token
```

- **P** 负责 KV cache 生产（prefill + 传输），不生成 token
- **D** 负责解码（decode），KV 来自传输，不做本地 prefill
- 两套均使用 `cudagraph_mode: FULL_DECODE_ONLY`、`npugraph_ex: True`

---

## MTP 配置关键点

```json
{"method":"mtp","num_speculative_tokens":1,"enforce_eager":true}
```

**为什么 `enforce_eager: true`？**
PyTorch 2.11 AOTAutograd 编译器处理 MTP 推测模型时触发 `AssertionError: expected OutputCode`。`enforce_eager` 跳过推测模型的 torch.compile（主模型 npugraph_ex 不受影响）。此前曾误关 npugraph_ex——该错误方向已纠正。

---

## 故障排查速查表

| 症状 | 检查 | 修复 |
|:--|:--|:--|
| P/D 启动后退出 | `tail -50 /tmp/pd_test/prefill*.log` | 看 `Engine core initialization failed` 具体原因 |
| 健康检查超时 | `ps aux \| grep vllm` | 增大 launch.sh 循环次数 |
| Benchmark 异常 | 检查 P/D 是否真正分离 | 见下方「手工检查」 |
| AssertionError expected OutputCode | MTP enforce_eager 缺失 | 确认代码已 git pull 最新 |

### 手工检查 P/D 是否真正分离

```bash
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"dsv4","messages":[{"role":"user","content":"hi"}],"max_tokens":8}' > /dev/null
sleep 10
tail -20 /tmp/pd_test/decode*.log | grep "prompt throughput"
```
- `prompt throughput: 0.0 tokens/s` → ✅ PD 分离成功
- `prompt throughput: XXXX tokens/s`（非零）→ ❌ D 在本地做 prefill

---

## 关键结果速览

| 指标 | No-MTP (4K/1K c=1) | MTP (4K/1K c=1) | MTP 改善 |
|:--|:--|:--|:--|
| TTFT p50 | 326.6ms | 320.6ms | −1.8% |
| TPOT p50 | 42.0ms | 23.4ms | **−44.3%** |
| 吞吐 (tok/s) | 23.64 | 41.92 | **+77.3%** |

详见 `SUMMARY.md` 和 `perf_opt/summary/pd_mtp_bench_summary.md`。

---

## 清理

```bash
pkill -f "vllm serve"          # 停止所有 P/D 进程
rm -f /tmp/pd_test/*.log       # 清理日志
rm -f /tmp/pd_bench/*.json     # 清理 bench 结果
```
- 远程工作目录：`/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test`
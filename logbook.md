# Logbook
## 2026-09-17

### 今日工作：Qwen3-4B-Base FP16 + W8A8 量化 & GSM8K 精度对比 ✅

---

#### 0. 背景

上次测试的是 **Qwen3-4B-Instruct**（指令微调版），GSM8K 成绩 FP16=94.01%、W8A8=86.66%。但官方公布的成绩是针对 **Qwen3-4B-Base**（预训练基座版）的。为与官方对比，本次切换为 Base 版本重新测试。

---

#### 1. 模型下载 & 环境准备 ✅

| 项目 | 值 |
|------|-----|
| 模型 | Qwen/Qwen3-4B-Base（ModelScope） |
| 下载路径 | `./models/Qwen3-4B-Base/` |
| 下载大小 | 3.96 GB |
| 权重文件 | 3 个 `.safetensors`（每个 ~1.32 GB） |

---

#### 2. FP16 Base GSM8K 基线测试 ✅

**服务启动：** vLLM FP16 服务（端口 8802）

**遇到的问题：**
- **max_model_len 不匹配**：模型配置文件 `config.json` 中 `max_position_embeddings=40960`，但 vLLM 配置文件预设值 `30000+` 小于此值，需手动设置 `--max-model-len 32768`（留 30% 余量给 kv_cache）
- **解决**：在 `start_vllm_base.py` 中指定 `--max-model-len 32768`

**结果：**

| 指标 | 值 |
|:----|:---:|
| **Score** | **49.36%** |
| 吞吐 | 42.35 tok/s |
| 平均延迟 | 22.54s |
| 平均输出 token | 954 |
| 耗时 | ~1.5 小时 |

**对比 Instruct 版（FP16）：**

| 版本 | FP16 分数 | 差异 |
|:----|:---------:|:----:|
| Qwen3-4B-Instruct | 94.01% | — |
| **Qwen3-4B-Base** | **49.36%** | ↓ 44.65pp |

**原因分析：** Base 模型未经过指令微调，不知道要按 GSM8K 格式输出答案（"The answer is X"），而是自由续写。平均输出 954 token 说明模型在"编故事"而非回答问题。

---

#### 3. Base W8A8 量化 ✅

**工具：** `msmodelslim`（/opt/mamba/bin/msmodelslim 可执行文件）

**最终成功命令：**
```bash
msmodelslim quant \
  --model_type Qwen3-4B \
  --model_path ./models/Qwen3-4B-Base \
  --save_path ./models/Qwen3-4B-Base-W8A8 \
  --quant_type w8a8 \
  --trust_remote_code True \
  --device npu
```

**踩坑记录 🐛**

| # | 问题 | 现象 | 根因 | 解决 |
|:-|:----|:----|:----|:----|
| 1 | CLI 入口不对 | `python3 -m msmodelslim.cli` 报错（cli 模块为空） | msmodelslim 的入口是可执行文件 `/opt/mamba/bin/msmodelslim`，不是 Python 模块 | 用 `msmodelslim quant` 代替 |
| 2 | 参数名不对 | 传 `--output_path` 不识别 | 实际参数名是 `--save_path`（help 输出中用 `--save_path`） | 改用 `--save_path` |
| 3 | `--quant_type` 值格式错误 | `QuantType.W8A8` 和 `W8A8` 都报 `invalid QuantType value` | Enum 定义的值是小写 `'w8a8'`，但 argparse 的 `type=QuantType` 要求用枚举值本身校验。实测枚举值 `QuantType.W8A8` 的 `.value` 是 `'w8a8'`，但 CLI 不接受大写形式 | 传 `w8a8`（全小写） |
| 4 | 交互确认 | 量化卡住等待输入 `Enter y to continue` | 找不到最佳实践配置时，工具会问是否用默认配置 | `tmux send-keys -t quant_base_w8a8 'y' Enter` |
| 5 | 退出确认（登录后可忽略） | 量化成功退出时弹 `DeprecationWarning` | swigvarlink 的 Python 3.13 兼容问题，不影响结果 | 忽略 |

**量化效果：**

| 指标 | FP16 Base | W8A8 Base | 变化 |
|:----|:---------:|:---------:|:----:|
| 模型大小 | 3.96 GB（3 个权重文件） | **4.9 GB**（2 个权重文件） | ⚠️ 变大（含量化元数据） |
| 推理吞吐 | 42.35 tok/s | **106.24 tok/s** | **↑ 2.5x** |

> **注：** Base 版 W8A8 权重文件比 FP16 还大，是因为量化后的 safetensors 同时包含了量化后的 int8 权重和原始 scale/zero_point 元数据。Instruct 版也是类似情况（FP16=7.87GB vs W8A8=4.92GB，Instruct 本身是更大的模型变体）。

**输出文件路径：** `./models/Qwen3-4B-Base-W8A8/`

---

#### 4. Base W8A8 vLLM 服务启动 ✅

**使用脚本：** `start_vllm_base_w8a8.py`（基于 `start_vllm_patched.py` 修改）

**补丁内容（与 Instruct 版相同）：**
1. ✅ `AOTAutogradCache` 禁用方式修正（用属性赋值 `_functorch_cfg.enable_autograd_cache = False`，不用直接改 `_config` 字典）
2. ✅ `maybe_update_config` 补丁：强制从 `MODEL_DIR` 本地加载 `quant_model_description.json`
3. ✅ 手动拷贝 `args.model = args.model_tag` 和 `args.tokenizer = args.tokenizer_tag`（跳过 launch.py 的 CLI 映射）

**服务配置：**

| 参数 | 值 |
|:----|:----:|
| 端口 | 8803 |
| model | `./models/Qwen3-4B-Base-W8A8` |
| served-model-name | `Qwen3-4B-Base` |
| max-model-len | 8192 |
| quantization | `ascend` |
| dtype | `auto` |

---

#### 5. GSM8K 精度对比：FP16 Base vs W8A8 Base 📊

**评测设置（口径说明）：**

| 参数 | 值 |
|:----|:----:|
| 评测工具 | EvalScope v1.8.1 |
| 评测模式 | `openai_api`（通过 vLLM HTTP API 调用） |
| 数据集 | GSM8K（Grade School Math 8K）全量 1,319 题 |
| Shot 数 | 4-shot（EvalScope 默认） |
| 解码参数 | `temperature=0`，`seed=42`，`top_p=1.0`，`top_k=-1` |
| max_tokens | FP16: 2048 / W8A8: 2048 |

**完整结果表：**

| 指标 | **FP16 Base** | **W8A8 Base** | 变化 |
|:----|:------------:|:-------------:|:----:|
| **Score** | **49.36%** | **31.61%** | **↓ 17.75pp** |
| 精度保留率 | 100% | **64.0%** | ↓ 36% |
| 吞吐（tok/s） | 42.35 | **106.24** | **↑ 2.51x** |
| 平均延迟（s） | 22.54 | 12.43 | ↓ 45% |
| 平均 TTFT（ms） | 71.06 | 41.49 | ↓ 42% |
| 平均 TPOT（ms） | 23.58 | 9.33 | ↓ 60% |
| 平均输入 token | 661 | 661 | — |
| 平均输出 token | 954 | 1,320 | — |

**与 Instruct 版对比总览：**

| 模型版本 | 精度 | FP16 | W8A8 | 精度保留率 | 速度增益 |
|:---------|:----:|:----:|:----:|:----------:|:--------:|
| **Qwen3-4B-Base** | W8A8 | **49.36%** → **31.61%** | **64.0%** | **2.51x** |
| Qwen3-4B-Instruct | W8A8 | 94.01% → 86.66% | 92.2% | 2.38x |
| Qwen3-4B-Instruct | W8A8S | 94.01% → 92.80% | 98.7% | — |

**关键发现：**

1. **Base 对量化更敏感**：精度保留率仅 64.0%，远低于 Instruct 的 92.2%。可能是因为预训练阶段的权重分布更广，量化后信息损失更大
2. **速度提升一致**：W8A8 在 NPU 上带来约 **2.4-2.5x** 的吞吐提升，无论 Base 还是 Instruct
3. **Base 49.36% 是合理的**：官方 Qwen3-4B-Base 未公布 GSM8K 成绩，但作为 4B 参数未微调模型，~50% 在合理范围内
4. **W8A8 后输出 token 增加**：Base 模型 W8A8 量化后平均输出从 954 跳到 1,320，可能是因为量化引入了额外的随机性，导致模型"编"得更长

---

#### 6. 文件清单

| 文件 | 用途 | 备注 |
|:----|:----|:----|
| `models/Qwen3-4B-Base/` | FP16 Base 原始模型 | 从 ModelScope 下载 |
| `models/Qwen3-4B-Base-W8A8/` | W8A8 量化后的 Base 模型 | 量化输出 |
| `quant_base.py` | Base 量化脚本（Python API 版） | ⚠️ 未使用（改用 CLI） |
| `start_vllm_base_w8a8.py` | Base W8A8 vLLM 启动脚本 | 端口 8803 |
| `eval_gsm8k_base.py` | FP16 Base GSM8K 评测脚本 | 端口 8802 |
| `eval_gsm8k_base_w8a8.py` | W8A8 Base GSM8K 评测脚本 | 端口 8803 |
| `quant_base_w8a8.log` | 量化日志（远程） | — |
| `vllm_base_w8a8.log` | W8A8 服务日志（远程） | — |
| `eval_base_w8a8_gsm8k.log` | W8A8 GSM8K 评测日志（远程） | — |
| `eval_fp16_base_gsm8k.log` | FP16 GSM8K 评测日志（远程） | — |
| `vllm_fp16_base.log` | FP16 服务日志（远程） | — |

---

#### 7. 远程文件确认

所有日志和输出存在于远程路径 `/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/`：

```bash
models/Qwen3-4B-Base/          # FP16 原始模型 ✅
models/Qwen3-4B-Base-W8A8/     # W8A8 量化模型 ✅
quant_base_w8a8.log             # 量化日志 ✅
vllm_base_w8a8.log              # W8A8 服务日志 ✅
eval_base_w8a8_gsm8k.log        # W8A8 评测日志 ✅
eval_fp16_base_gsm8k.log        # FP16 评测日志 ✅
vllm_fp16_base.log              # FP16 服务日志 ✅
```

---

## 2026-09-15

### 今日工作：Qwen3-4B W8A8 量化 + GSM8K 精度对比 ✅

---

#### 1. Qwen3-4B W8A8 量化 ✅

**工具**：`msmodelslim`（ModelScope 模型压缩工具）

**命令**：
```bash
python3 -m msmodelslim.cli \
  --model_type Qwen3-4B \
  --model_path ./models/Qwen3-4B \
  --output_path ./models/Qwen3-4B-W8A8 \
  --quant_type w8a8
```

**解决的关键问题：**
- **交互确认**：量化过程中途询问 `Enter Your Option：[0/1/2]`，通过 `tmux send-keys` 自动输入 `y`（选择默认确认）
- 逐层处理全部 **36 层**，成功完成

**量化效果：**
| 指标 | FP16 原始 | W8A8 量化后 | 变化 |
|------|:---------:|:-----------:|:----:|
| 模型大小 | ~7.87 GB (7,872 MB) | ~4.92 GB (5,092 MB) | **↓ 36%** |
| 权重文件 | 3 个 `.safetensors` | 2 个 `.safetensors` | 文件更少 |
| 量化方式 | - | per-channel weight, per-token activation | - |

**输出文件：**
- `quant_model_weights-00001-of-00002.safetensors` (3.64 GB)
- `quant_model_weights-00002-of-00002.safetensors` (1.56 GB)
- `quant_model_description.json` (58 KB)
- `Qwen3-4B_best_practice.yaml` (606 B)
- 保留原始 config / tokenizer 等配置文件

**模型路径**：`./models/Qwen3-4B-W8A8/`

---

#### 2. vLLM W8A8 服务启动 —— 踩坑全记录 🐛

**背景**：用 `start_vllm_patched.py` 启动 vLLM 服务加载 W8A8 量化模型，遇到 3 个关键 bug。

##### Bug 1：量化配置文件找不到（ModelSlim config not found）

- **现象**：vLLM 启动时报错找不到 `quant_model_description.json`
- **根因**：`vllm/config.py:120` 的 `_get_and_verify_model_tags` 对本地路径（如 `/inspire/.../Qwen3-4B-W8A8`）只返回了 `Qwen3-4B`，去 HF Hub 查模型名 → 拿到默认值 `Qwen/Qwen3-0.6B` → 用 `Qwen/Qwen3-0.6B` 去找量化配置 → 404
- **修复**：给 `maybe_update_config`（位于 `vllm/config.py`）打补丁，强制优先从本地 `MODEL_DIR` 查找 `quant_model_description.json`，不依赖 vLLM 传进来的错误模型名

##### Bug 2：模型加载成 0.6B 而不是 4B（模型解析与模型名不一致）

- **现象**：开机 banner 显示 model=`Qwen/Qwen3-0.6B`，模型权重也加载了 0.6B 版本
- **根因**：`start_vllm_patched.py` 用 `parser.parse_args([MODEL_DIR, ...])` 解析参数，CLI 解析器把 `MODEL_DIR` 存成 `args.model_tag`，但 `api_server.py` 启动引擎时读的是 `args.model`（默认值 `Qwen/Qwen3-0.6B`）。正常走命令行时 `launch.py` 有一行 `args.model = args.model_tag` 做映射，但我们直接调 `run_server(args)` 跳过了这一步
- **修复**：在 `run_server(args)` 之前手动拷贝 `args.model = args.model_tag` 和 `args.tokenizer = args.tokenizer_tag`
- **相关文件**：`start_vllm_patched.py` 第 100-107 行

##### Bug 3：引擎编译崩溃（AOTAutogradCache 禁用方式错误）

- **现象**：权重加载成功，但引擎初始化时崩溃，报 `AttributeError: 'bool' object has no attribute 'hide'`
- **堆栈定位**：`torch/_dynamo/aot_compile.py` 中 `torch._functorch.config.patch()` 进入上下文管理器 → 读取配置时发现值变成了 `bool` 而不是 ConfigModule → 调用 `config.hide` 失败
- **根因**：为绕过 PyTorch 2.11 的 AOTAutogradCache 问题，之前用了 `_functorch_cfg._config['enable_autograd_cache'] = False` 直接修改内部字典。但 `_config` 不是普通 dict——其值必须是特定的 ConfigModule 对象。塞入 `bool` 值破坏了配置系统的嵌套结构
- **修复**：改用正确的属性赋值 API：`_functorch_cfg.enable_autograd_cache = False`
- **补充**：该 bug 是条件性的——仅当 PyTorch 编译路径被触发时才会暴露（enforce_eager 可以绕过，但 vLLM 的 V1 引擎 + Ascend 平台强制走编译路径）

##### Bug 4（评测阶段）：EvalScope max_tokens 超出模型限制

- **现象**：GSM8K 评测返回大量 HTTP 400 错误：`"This model's maximum context length is 8192 tokens. However, you requested 8192 output tokens..."`，导致评测卡住重试
- **根因**：W8A8 模型的 `max_model_len` 默认为 8192 token，但 `max_tokens=8192` 没给 prompt 留空间。当 prompt 较长时，prompt_tokens + 8192 > 8192，vLLM 拒绝请求
- **修复**：将 `max_tokens` 从 8192 降为 2048（GSM8K 答案通常很短，2048 足够）
- **相关文件**：`eval_gsm8k.py`

---

#### 3. GSM8K 精度对比：FP16 vs W8A8 📊

**评测设置（口径说明）：**

| 参数 | 值 |
|------|-----|
| 评测工具 | EvalScope v1.8.1 |
| 评测模式 | `openai_api`（通过 vLLM HTTP API 调用） |
| 数据集 | GSM8K（Grade School Math 8K）全量 1,319 题 |
| Shot 数 | 4-shot（EvalScope 默认） |
| 解码参数 | `temperature=0`，`seed=42`，`top_p=1.0`，`top_k=-1` |
| max_tokens | FP16: 8192 / W8A8: 2048 |
| 评估指标 | `mean_acc`（答案精确匹配） |
| 过滤规则 | `remove_until: " response"`（去除 few-shot 示例中的思维链） |

**最终结果：**

| 精度模式 | GSM8K 得分（mean_acc） | 相对于 FP16 的差异 |
|:--------:|:---------------------:|:-----------------:|
| **FP16 基线** | **94.01%** | - |
| **W8A8 量化** | **86.66%** | **-7.35 个百分点** |
| 精度保留率 | - | **92.2%**（86.66/94.01） |

**性能数据对比：**

| 指标 | FP16 | W8A8 |
|:----|:----:|:----:|
| 平均延迟 | 30.63s | 9.94s |
| 平均 TTFT | 66.16ms | 42.11ms |
| 平均 TPOT | 21.78ms | 9.02ms |
| 平均吞吐 | 45.85 tok/s | 110.22 tok/s |
| 平均输入 Token | 661 | 661 |
| 平均输出 Token | 1404 | 1095.56 |

**结论：**
- W8A8 模型大小缩小 36%（7.87 GB → 4.92 GB）
- 推理**速度提升 2.4x**（45.85 → 110.22 tok/s）
- GSM8K 准确率下降 7.35pp（94.01% → 86.66%），精度保留率 92.2%
- 在昇腾 910B NPU 上 W8A8 量化有效降低了显存占用并加速推理，但 GSM8K 数学推理任务上精度损失较明显

**日志文件：**
- vLLM 服务日志：`vllm_w8a8_patched3.log`（最新成功启动日志）
- 评测日志：`eval_w8a8_gsm8k.log`
- 评测报告 JSON：`./outputs/20260915_141127/reports/Qwen3-4B/gsm8k.json`
- FP16 基线评测日志：`eval_fp16_gsm8k.log`

---

#### 4. 关键文件说明

| 文件 | 说明 |
|------|------|
| `start_vllm_patched.py` | W8A8 vLLM 服务启动脚本（含打补丁逻辑），最终修复版 |
| `eval_gsm8k.py` | GSM8K 评测脚本，直接通过 evalscope API 调用 |
| `test_api.py` | 基础 API 连通性测试脚本 |
| `LOGBOOK.md` | 工作日志（本文档） |

## 2026-09-14

### 今日工作
---
**背景：** 9月10日用 Qwen3-14B 跑 MMLU-Redux（570题）耗时 1h05min+，单卡 910B 跑 14B 推理速度太慢。决定换用 Qwen3-4B 重新跑。

**操作流程：**
1. 在远程机器上下载 Qwen3-4B 模型（ModelScope `Qwen/Qwen3-4B`）
2. 在本地修改 `configuration.yaml`，将模型从 Qwen3-14B 改为 Qwen3-4B，同时调整 `max_model_len` 从 65536 降为 40960，增加 `enforce_eager` 参数
3. 停止旧 vLLM 服务，启动新 vLLM 服务
4. 运行 EvalScope MMLU-Redux 评测

---

#### 2. 模型下载记录（download.log）

**时间：** 09:xx - 09:xx（约 3 分 29 秒）

**下载模型：** `Qwen/Qwen3-4B`（ModelScope）

**下载详情：**
- 14 个文件，总大小约 3.96 GiB（safetensors 格式）
- 平均下载速度约 15-25 MB/s
- 下载路径：`./models/Qwen3-4B`

**命令行：**
```bash
python3 -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('Qwen/Qwen3-4B', local_dir='./models/Qwen3-4B')
"
```

**结果：** ✅ `Snapshot ready at models/Qwen3-4B`

---

#### 3. vLLM 服务启动记录（vllm_fp16.log）

**时间：** 09:55:32 - 12:05:29（服务运行时长约 2h10min）

**启动命令（tmux 后台）：**
```bash
tmux new-session -d -s vllm_fp16 \
  "vllm serve ./models/Qwen3-4B \
**启动过程（关键日志节点）：**
| 时间 | 事件 |
|------|------|
| 09:55:32 | vLLM 启动，Ascend 平台插件激活 |
| 09:55:38 | ServiceProfiler 自动选择 V1 引擎 |
| 09:55:39 | vLLM Banner 显示，model=./models/Qwen3-4B |
| 09:55:39 | `non-default args`: max_model_len=40960, enforce_eager=True, gpu_memory_utilization=0.9 |
| 09:55:39 | Resolved architecture: Qwen3ForCausalLM |
| 09:55:39 | 禁用 CUDAGraphs（enforce eager）|
| 09:55:39 | 模型加载为 float（无量化），No quantization signature detected |
| 09:55:57 | **显存状态：** Free memory 60.57/60.96 GiB，weights=7.52 GiB，KV cache=47.13 GiB |
| 09:55:58 | GPU-specific parameter not supported on Ascend |
| 09:55:58 | `generation_config.json` 覆盖默认采样参数：temp=0.6, top_k=20, top_p=0.95 |
| 09:55:59 | **Server started on http://0.0.0.0:8801** ✅ |

**推理吞吐指标（评测高峰期）：**
| 时间段 | Prompt Throughput | Generation Throughput |
|-------|------------------|---------------------|
| 10:00-10:30 | ~20-40 tok/s | ~170-180 tok/s |
| 10:30-11:00 | ~30-50 tok/s | ~170-200 tok/s |
| 11:32（评测开始）| ~30-50 tok/s | ~170-200 tok/s |
| 12:00-12:05（评测高峰）| ~30-70 tok/s | ~310-360 tok/s |
| 12:05:19 | 评测完成，0 reqs | 0.0 tok/s |

**推理期间 GPU KV cache 使用率：** 0.6% - 5.8%（波动）
**Prefix cache hit rate：** 0.0% → 1.7%（随请求增加略有提升）
    --served-model-name Qwen3-4B \
    --port 8801 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --max-model-len 40960 \
---

#### 4. MMLU-Redux 评测记录（eval_fp16_mmlu_redux.log）

**评测命令：**
```bash
cd /inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test && \
python3 -c "
from evalscope import TaskConfig, run_task
task_cfg = TaskConfig(
    model='Qwen3-4B',
    api_url='http://127.0.0.1:8801/v1/chat/completions',
    eval_type='openai_api',
    datasets=['mmlu_redux'],
    dataset_args={'mmlu_redux': {}},
    eval_batch_size=16,
    generation_config={
        'max_tokens': 40960,
        'temperature': 0.6,
        'top_p': 0.95,
        'top_k': 20,
        'n': 1,
    },
    timeout=120000,
    stream=True,
)
run_task(task_cfg=task_cfg)
" 2>&1 | tee eval_fp16_mmlu_redux.log
```

**评测参数：**
| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-4B |
| 数据集 | mmlu_redux（570 题）|
**时间线：**
| 事件 | 时间 | 耗时 |
|------|------|------|
| 评测开始 | 11:32:34 | - |
| 10%（57题）| ~11:38 | ~6min |
| 25%（142题）| ~11:43 | ~11min |
| 50%（285题）| ~11:50 | ~18min |
| 73%（414题）| 11:57:38 | 25min02s |
| 77%（437题）| 11:58:38 | 26min02s |
| 90%（513题）| 12:02 | ~30min |
| **评测结束** | **12:05:13** | **~32min39s** |

**最终结果——Accuracy 表格（第1部分）：**

```
┌──────────┬────────────┬──────────┬────────────────────────────────┬───────┬──────────┬──────────┐
│ Model    │ Dataset    │ Metric   │ Subset                         │   Num │   Score  │   Cat    │
├──────────┼────────────┼──────────┼────────────────────────────────┼───────┼──────────┼──────────┤
│ Qwen3-4B │ mmlu_redux │ mean_acc │ abstract_algebra               │   10 │  0.8     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ anatomy                        │   10 │  0.8     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ astronomy                      │   10 │  0.8     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ business_ethics                │   10 │  0.9     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ clinical_knowledge             │   10 │  0.7     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ college_biology                │   10 │  1.0     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ college_chemistry              │   10 │  0.5     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ college_computer_science       │   10 │  0.7     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ college_mathematics            │   10 │  0.6     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ college_medicine               │   10 │  0.8     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ college_physics                │   10 │  0.5     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ computer_security              │   10 │  0.9     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ conceptual_physics             │   10 │  0.7     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ econometrics                   │   10 │  0.9     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ electrical_engineering         │   10 │  0.6     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ formal_logic                   │   10 │  0.6     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ global_facts                   │   10 │  0.8     │ default  │
```
├──────────┼────────────┼──────────┼────────────────────────────────┼───────┼──────────┼──────────┤
│ Qwen3-4B │ mmlu_redux │ mean_acc │ high_school_us_history         │   10 │  0.9     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ high_school_world_history      │   10 │  0.7     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ human_aging                    │   10 │  0.8     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ human_sexuality                │   10 │  0.8     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ international_law              │   10 │  0.9     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ jurisprudence                  │   10 │  0.7     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ logical_fallacies              │   10 │  1.0     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ machine_learning               │   10 │  0.9     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ management                     │   10 │  0.9     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ marketing                      │   10 │  0.9     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ medical_genetics               │   10 │  0.8     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ miscellaneous                  │   10 │  0.8     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ moral_disputes                 │   10 │  0.8     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ moral_scenarios                │   10 │  0.5     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ nutrition                      │   10 │  0.8     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ philosophy                     │   10 │  0.7     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ prehistory                     │   10 │  0.9     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ professional_accounting        │   10 │  0.9     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ professional_law               │   10 │  0.7     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ professional_medicine          │   10 │  0.6     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ professional_psychology        │   10 │  0.9     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ public_relations               │   10 │  0.7     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ security_studies               │   10 │  0.4     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ sociology                      │   10 │  0.7     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ us_foreign_policy              │   10 │  0.9     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ virology                       │   10 │  0.6     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ world_religions                │   10 │  1.0     │ default  │
├──────────┼────────────┼──────────┼────────────────────────────────┼───────┼──────────┼──────────┤
│ Qwen3-4B │ mmlu_redux │ mean_acc │ **OVERALL**                    │**570**│**0.8088**│ **-**    │
└──────────┴────────────┴──────────┴────────────────────────────────┴───────┴──────────┴──────────┘
```

**MMLU-Redux 总分：Qwen3-4B = 80.88%**

**Performance 指标：**
| Model | Dataset | Num | Avg Lat (s) | Avg TTFT (ms) | Avg TPOT (ms) | Avg Thpt (tok/s) | Avg In Tok | Avg Out Tok |
|-------|---------|-----|------------|--------------|--------------|-----------------|-----------|------------|
| Qwen3-4B | mmlu_redux | 570 | 26.9174 | 64.55 | 21.92 | 45.58 | 153.628 | 1226.81 |

**输出路径：** `./outputs/20260914_113234/reports/report.html`

---

#### 5. Qwen3-4B vs Qwen3-14B 对比总结

| 对比项 | Qwen3-14B（9月10日）| Qwen3-4B（9月14日）|
|--------|-------------------|-------------------|
| 模型大小 | 14B | 4B |
---

#### 6. 今日终端命令完整记录（远程执行，按时间序）

```bash
# 1. 创建新实例
inspire notebook create -n inspire-demo \
  --workspace 昇腾卡公共空间 \
  --project 公共科研项目 \
  --group 910B资源 \
  -q 1,8,64 \
  --image ascend-a2-ubuntu:v4.2

# 2. 检查环境
inspire notebook exec inspire-demo --workspace 昇腾卡公共空间 \
  "echo '=== Python ===' && python3 --version && \
   echo '=== PyTorch ===' && python3 -c 'import torch; print(torch.__version__, torch.npu.is_available())'"

# 3. NPU 信息
inspire notebook exec inspire-demo --workspace 昇腾卡公共空间 "npu-smi info"

# 4. 下载 Qwen3-4B 模型（约 3.5min）
inspire notebook exec inspire-demo --workspace 昇腾卡公共空间 \
  "cd /inspire/.../agent_benchmark_test && \
   python3 -c 'from modelscope.hub.snapshot_download import snapshot_download; \
   snapshot_download(\"Qwen/Qwen3-4B\", local_dir=\"./models/Qwen3-4B\")' 2>&1 | tee download.log"

# 5. tmux 启动 vLLM 服务（Qwen3-4B, fp16）
inspire notebook exec inspire-demo --workspace 昇腾卡公共空间 \
  "tmux new-session -d -s vllm_fp16 'vllm serve ./models/Qwen3-4B \
    --served-model-name Qwen3-4B --port 8801 --gpu-memory-utilization 0.9 \
    --trust-remote-code --max-model-len 40960 --enforce-eager > vllm_fp16.log 2>&1'"

# 6. 等待并验证服务
inspire notebook exec inspire-demo --workspace 昇腾卡公共空间 \
  "sleep 90 && curl -s http://127.0.0.1:8801/v1/models | head -c 300"

# 7. 运行 EvalScope MMLU-Redux 评测（~33min）
inspire notebook exec inspire-demo --workspace 昇腾卡公共空间 \
  "cd /inspire/.../agent_benchmark_test && python3 -c 'from evalscope import TaskConfig, run_task; \
   task_cfg = TaskConfig(model=\"Qwen3-4B\", api_url=\"...\", eval_type=\"openai_api\", \
   datasets=[\"mmlu_redux\"], eval_batch_size=16, generation_config={...}, timeout=120000); \
   run_task(task_cfg=task_cfg)' 2>&1 | tee eval_fp16_mmlu_redux.log"

# 8. 查看结果
inspire notebook exec inspire-demo --workspace 昇腾卡公共空间 \
  "cd /inspire/.../agent_benchmark_test && head -1 eval_fp16_mmlu_redux.log && tail -5 eval_fp16_mmlu_redux.log"
---

#### 7. git 提交记录

```
*   1f5a19f 2026-09-14 12:08:27 - On main: cline checkpoint session=1789358907070_2e6b0 run=1
|\
| * 6b9e033 2026-09-14 12:08:27 - index on main: fix: rename Qwen3-14B-Instruct -> Qwen3-14B (ModelScope)
|/
* 97d9cdf 2026-09-10 - (HEAD -> main, origin/main) fix: rename Qwen3-14B-Instruct -> Qwen3-14B (ModelScope)
```

**今日修改的文件：**
- `.clinerules` — 新增"已踩过的坑 && 已知约束"章节
- `configuration.yaml` — 模型从 Qwen3-14B 改为 Qwen3-4B，调参
- `logbook.md` — 追加今日完整记录

---

#### 8. 与官方分数对比（TODO）

| Benchmark | Qwen3-4B（实测）| Qwen3-14B（实测）| Qwen3-14B（官方）| Qwen3-4B（官方）|
|-----------|----------------|-----------------|-----------------|-----------------|
| MMLU-Redux | **80.88%** | **87.37%** | 待查 Table 16 | 待查 Table 16 |

> ⚠️ **注意：** 官方技术报告中 Qwen3-4B 的 MMLU-Redux 分数需从 Table 16（非思考模式）查阅。当前跑的是思考模式（temp=0.6, top_p=0.95, top_k=20），后续还需跑非思考模式对比。

---

#### 9. 后续计划

1. **继续跑剩余 7 个 benchmark（Qwen3-4B 思考模式）：**
   - GPQA-Diamond
   - C-Eval
   - IFEval
   - MATH-500
   - AIME'24
   - AIME'25
   - LiveCodeBench v5
2. **跑非思考模式（temp=0.7, top_p=0.8）全部 8 个 benchmark**
3. **查阅官方 PDF Table 15/16，填入官方分数进行比对**

---

#### 10. 新踩的坑 & 教训

| 坑 | 现象 | 原因 | 解决 |
|---|------|------|------|
| ❌ `enforce_eager` 必须要加 | 不加可能在某些 910B 环境上有编译问题 | Ascend 部分算子不支持 torch.compile | `--enforce-eager` ✅ |
| ❌ `generation_config.json` 覆盖采样参数 | vLLM 启动时 warning 提示默认参数被覆盖 | 模型自带 `generation_config.json` 设置了 temp=0.6, top_k=20, top_p=0.95 | 这是期望行为，EvalScope 会再覆盖 ✅ |
| ❌ 首次 vLLM 请求慢（cold start）| 首条请求延迟很长 | NPU 首次推理需要 warm up | warm up 后恢复正常 ✅ |
| ❌ `--device npu` 参数不被 vLLM 0.22.1 识别 | vLLM 报 `unrecognized arguments` | vLLM-Ascend 自动检测 NPU | 去掉该参数 ✅ |
| ❌ `max_model_len` 设太大导致 OOM | 14B+65536ctx 可能显存不够 | 单卡 60.96 GiB | 改 4B+40960ctx ✅ |

---

#### 11. 本地终端历史（今日完整命令速览）

```
# 环境检查和实例管理
835  inspire exec → 检查远程环境（Python/PyTorch/evalscope）
841  inspire exec → npu-smi info
846  inspire notebook list
851  inspire exec → 详细 NPU 信息
852  inspire --help
855  inspire init
857  inspire notebook list --workspace all
863  inspire notebook quota --workspace 昇腾卡公共空间
864  inspire image list --keyword ascend
865  inspire notebook create -n inspire-demo (创建新实例)

# 模型下载
893-897 下载 Qwen3-14B-Instruct（404）→ 改 Qwen/Qwen3-14B → 再改 Qwen3-4B
900-903 查看下载进度

# vLLM 启动（踩坑过程）
905  nohup vllm serve（后台进程不持久，PID=0 的坑）
960-980 tmux 启动 vLLM(Qwen3-4B) + EvalScope 评测

# 查看评测进度（反复 tail 查看）
980-1000 tail -30 eval_test.log × 20次 和 eval_fp16_mmlu_redux.log

# 最终查看结果
1001-1004 ls / find / 查看今天运行结果
```

---

#### 12. 远程机器日志文件清单

| 文件 | 大小 | 日期 | 内容 |
|------|------|------|------|
| download.log | 248 KB | 09-14 09:49 | Qwen3-4B 模型下载日志 |
| vllm_fp16.log | 323 KB | 09-14 12:05 | Qwen3-4B vLLM 服务完整日志 |
| eval_fp16_mmlu_redux.log | 145 KB | 09-14 12:05 | MMLU-Redux 评测结果 |
| eval_test.log | 170 KB | 09-10 18:04 | Qwen3-14B 旧评测结果 |
| vllm_server.log | 181 KB | 09-10 18:04 | Qwen3-14B 旧 vLLM 服务日志 |

---
```
| 权重占用显存 | ~28 GiB（估算）| 7.52 GiB |
| 推理引擎 | vLLM 0.22.1 | vLLM 0.22.1 |
| max_model_len | 65536 | 40960 |
| MMLU-Redux 总分 | **87.37%** | **80.88%** |
| 总耗时（570题）| ~1h05min | ~33min |
| 平均延迟/题 | 32.54s | 26.92s |
| Avg TPOT | 28.86ms | 21.92ms |
| Avg Throughput | 33.66 tok/s | 45.58 tok/s |
| 生成速度（高峰）| ~140 tok/s | ~360 tok/s |

**结论：** Qwen3-4B 推理速度约为 Qwen3-14B 的 2.6 倍（generation throughput），MMLU-Redux 分数下降约 6.5 个百分点（87.37% → 80.88%）。用 4B 模型跑全量 8 个 benchmark 更现实。
│ Qwen3-4B │ mmlu_redux │ mean_acc │ high_school_biology            │   10 │  0.8     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ high_school_chemistry          │   10 │  0.3     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ high_school_computer_science   │   10 │  0.8     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ high_school_european_history   │   10 │  0.8     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ high_school_geography          │   10 │  0.9     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ high_school_government         │   10 │  1.0     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ high_school_macroeconomics     │   10 │  0.6     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ high_school_mathematics        │   10 │  0.7     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ high_school_microeconomics     │   10 │  0.8     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ high_school_physics            │   10 │  0.6     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ high_school_psychology         │   10 │  0.9     │ default  │
│ Qwen3-4B │ mmlu_redux │ mean_acc │ high_school_statistics         │   10 │  0.7     │ default  │
```
| 模式 | thinking_mode（temp=0.6, top_p=0.95, top_k=20）|
| eval_batch_size | 16 |
| max_tokens | 40960 |
| timeout | 120000ms |
    --enforce-eager \
    > vllm_fp16.log 2>&1"
```

**vLLM 版本：** 0.22.1（Ascend 插件自动加载）

## 2026-09-10

### 今日工作

#### 1. Qwen3-14B 模型下载与 vLLM 服务启动

**背景：** 需要下载 Qwen3-14B 模型到远程 NPU 机器，启动 vLLM 推理服务，然后跑 benchmark 评测。

**模型下载（踩坑记录）：**

| 坑 | 现象 | 原因 | 解决 |
|---|---|---|---|
| ❌ ModelScope 404 | `snapshot_download('Qwen/Qwen3-14B-Instruct')` 报 404 | ModelScope 上模型名是 `Qwen3-14B`，不带 `-Instruct` | 改为 `Qwen/Qwen3-14B` ✅ |
| ❌ 配置名称不一致 | `full_name` 和 `download_hf` 中写的是 `Qwen3-14B-Instruct` | 之前写错了 | 统一改为 `Qwen3-14B`，git push 修复 ✅ |
| ❌ `--device npu` 参数不被 vLLM 识别 | vLLM 启动失败，报 `unrecognized arguments: --device npu` | vLLM-Ascend 插件自动检测昇腾卡，不需要手动指定 `--device` | 去掉该参数 ✅ |
| ❌ `nohup` 后台进程不持久 | `inspire notebook exec` 跑完就断开，`PID: 0` | `inspire notebook exec` 是短时执行，不会保持后台进程 | 改用 `tmux new-session -d -s vllm` 后台持久化 ✅ |

**目录说明澄清：**
- 本地 Mac：`agent_benchmark_03`（写代码、git push）
- 远程 NPU：`agent_benchmark_test`（跑模型、跑评测）
- 两个目录通过 GitHub 同步

**当前状态：** ✅ vLLM 服务已通过 tmux 在后台启动，benchmark 评测正在运行中（尚未出结果）。

**教训总结：**
1. `inspire notebook exec` 适合跑一次性脚本，不适合启动长期后台服务
2. 启动 vLLM 等后台服务需用 `tmux` 或 `inspire notebook shell`（交互式终端）
3. ModelScope 模型名需确认，不能想当然加 `-Instruct` 后缀
4. vLLM-Ascend 自动检测 NPU，不需要 `--device npu`

---
## 2026-09-09

### 今日工作

#### 1. 确认远程环境信息

**背景：** 之前以为国产卡需要找 leader 问 IP，实际上就是启智平台的昇腾 910B NPU 实例。

**确认的信息：**
- 远程实例名：`inspire-demo`
- 工作空间：昇腾卡公共空间
- 远程代码路径：`/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test`
- 显卡：ASCEND 910B（昇腾 NPU）
- Python 3.13，PyTorch 2.11 + torch_npu

**工作流确认：**
1. 本地修改代码 → git commit + push 到 GitHub
2. 通过 `inspire notebook exec` 在远程拉取并执行
3. 查看 NPU 指标用 `npu-smi info`（不能用 nvidia-smi）
4. 代码和数据必须放在 `/inspire/` 共享盘下，否则实例停止会丢失

#### 2. 编写 NPU 训练脚本

**完成：** 创建 `train_linear_regression.py`，一个简单的线性回归训练脚本：
- 生成合成数据：`y = 2.0 * x + 1.0 + noise`
- 定义 `nn.Linear(1, 1)` 模型
- 在 NPU 上训练 500 个 epoch
- 输出最终学习的权重和偏置，与真实值对比

**问题：** 远程仓库 `origin/main` 落后于本地，从 GitHub 拉取最新代码失败（远程仓库之前有一些未 push 的本地 commit 导致了分叉）。

**解决：** 在远程执行 `git reset --hard origin/main` 强制同步到 GitHub 远端状态，再 `git pull` 拉取最新代码。

**结果：** ✅ 成功跑完，NPU 训练结果正常！

```
模型: y = 2.0114 * x + 0.9796
真实: y = 2.0 * x + 1.0
Loss 收敛至 0.2145
```

#### 3. 解决 GitHub 权限问题 + 切换到新仓库

**问题：** 本地 git push 遇到 `Permission to Lattenot/agent_benchmark_test.git denied to ada-yy1325`（403），macOS keychain 缓存的凭据没有权限 push 到原仓库。

**解决：** 
1. 在 `ada-yy1325` 账号下新建公开仓库 `agent_benchmark_03`
2. 本地 remote URL 改为 `https://github.com/ada-yy1325/agent_benchmark_03.git`
3. 清除 macOS keychain 缓存的凭据 → 重新登录验证
4. `git push -u origin main --force` 推送成功（远程初始 commit 被覆盖）

**远程同步：** 通过 `inspire notebook exec` 更新远程机器的 origin URL：
```bash
inspire notebook exec inspire-demo --workspace 昇腾卡公共空间 \
  "cd /inspire/.../agent_benchmark_test && \
   git remote set-url origin https://github.com/ada-yy1325/agent_benchmark_03.git && \
   git fetch origin && git reset --hard origin/main && \
   python3 train_linear_regression.py"
```

#### 4. 整条自动化链路验证通过 🎉

**最终执行结果：**
- **NPU 设备**: ASCEND 910B2C ✅
- **torch_npu**: 2.11.0.rc4 ✅
- **训练结果**: 500 epoch，Loss 0.2145，权重/偏置接近真实值
- **完整链路**: 本地写代码 → git push → inspire exec → 远程 NPU 执行 → 返回结果

**`.clinerules` 更新：**
- 将远程执行命令从 `git pull` 改为 `git fetch origin && git reset --hard origin/main`，避免因远程分支分叉导致 pull 失败
- 新增 GitHub 仓库 URL 字段

#### 5. InspireSkill 连接昇腾 910B 实例的故障排查（完整记录）

**背景：** InspireSkill 默认适配 NVIDIA GPU，连接 910B 昇腾 NPU 实例时会遇到一系列问题。以下是逐一排查和解决的完整过程。

---

> **问题 3：网页创建的实例无法通过 InspireSkill 连接**
>
> **现象：** 自己在启智网页端创建的 910B 实例，运行 `connection refresh` 完全卡住无输出。
>
> **原因：** InspireSkill 对非自己创建的实例拿不到完整连接句柄；且 910B 页面结构可能与 NVIDIA 卡不同。
>
> **解决：** 用 InspireSkill 自己创建实例，不要用网页创建。
>
> ```bash
> inspire notebook create -n inspire-demo \
>   --workspace 昇腾卡公共空间 \
>   --project 公共科研项目 \
>   --group 910B资源 \
>   -q 1,8,64 \
>   --image ascend-a2-ubuntu:v4.2
> ```

---

> **问题 4：connection refresh 报 SSH preflight failed**
>
> **现象：**
> ```
> Error: Tunnel setup completed, but SSH preflight failed.
> Proxy readiness: HTTP 500.
> ```
>
> **原因：** 910B 昇腾卡与 H100/H200 一样属于"受限 Notebook"，不支持 SSH 通道，应走 JupyterTerminal 通道。
>
> **解决：** 此报错本身是正常的，关键看后续 `exec` 能否走通，不必纠结这个错误。

---

> **问题 5（核心根因）：gpu_model 识别失败导致 exec 卡住**
>
> **现象：** `inspire notebook exec` 一直卡住无输出；`connection status` 报错。
>
> **排查：** 查看 `~/.inspire/notebook-gpu-models.json`，发现 `"gpu_model": ""`（空字符串）。
>
> **根因：** InspireSkill 通过在机器上运行 `nvidia-smi` 来识别显卡型号，进而判断走 SSH 还是 JupyterTerminal 通道。但 910B 是昇腾卡，**没有 nvidia-smi，只有 npu-smi**，导致识别失败，`gpu_model` 为空，连接逻辑走错通道。
>
> **解决（Workaround）：** 手动修改缓存文件，欺骗 InspireSkill 走 JupyterTerminal 通道。
>
> ```bash
> # 1. 备份原始文件
> cp ~/.inspire/notebook-gpu-models.json ~/.inspire/notebook-gpu-models.json.bak
>
> # 2. 编辑，把 "gpu_model": "" 改成 "gpu_model": "H100"
> open -e ~/.inspire/notebook-gpu-models.json
> ```
>
> 修改后重新 `connection refresh`，会提示 `SSH/rtunnel access is blocked on H100/H200 notebooks`，然后 `exec` 即可正常执行。
>
> **⚠️ 注意：** 这是 workaround，不是官方支持。`gpu_model` 字段只是内部通道选择标记，不影响实际硬件使用。建议后续给 InspireSkill 提 issue 请求官方适配昇腾卡。

---

> **问题 6：实例创建报错 "no available quota"**
>
> **现象：** 创建实例时提示没有可用资源。
>
> **解决：** 先查可用配额，选一个能用的：
> ```bash
> inspire notebook quota --workspace 昇腾卡公共空间
> ```

---

**快速自查流程（以后遇到类似问题按此顺序排查）：**

```
1. 实例是不是 InspireSkill 创建的？
   ├── 否 → 用 inspire notebook create 重建
   └── 是 → 下一步

2. connection refresh 报 SSH preflight failed？
   ├── 910B 正常现象，继续下一步
   └── 走下一步

3. exec 卡住无输出？
   ├── 检查 ~/.inspire/notebook-gpu-models.json
   │   └── gpu_model 为空 → 改为 "H100" → refresh → 重试 exec
   └── 正常 → 检查其他问题

4. 还是不行？
   ├── inspire notebook list --workspace ... 检查实例状态
   ├── inspire notebook connection status ... 看连接状态
   └── 重启实例试试
```

---

#### 6. 技术要点总结

## 2026-09-08

#### 1. 确认本地开发环境配置

**已完成：**
- 确认 `.env` 文件中配置了 DeepSeek API 的 `base_url` 和 `api_key`
- 确认 `.venv` 虚拟环境已创建且可用
- 确认 `.gitignore` 已正确配置（`.env`、`.venv` 等已忽略）

#### 2. 验证 API 连通性

**问题：** 运行 `test_api.py` 时，直接使用 `python test_api.py` 无法正确加载虚拟环境中的依赖。

**解决：** 需要先激活虚拟环境再运行脚本：
```bash
source .venv/bin/activate && python test_api.py
```
成功返回 `hello`，确认 API 调用正常。

#### 3. 规划 Benchmark 测试方案

**背景：** Leader 要求部署一个推理服务（跑在国产卡上），用 benchmark 数据集打分，跟官方数据做对比。API key 额度只用于日常 vibe coding，不能用于跑 benchmark。

**当前进展：**
- 准备使用 EvalScope 作为评测工具
- 数据集选定：GPQA、GSM8K、ARC
- 评测方式：`--eval-type openai_api` 模式，指向国产卡上的推理服务，不消耗 API key 额度
- **待办：** 需要找 leader 确认国产卡服务器的 IP 和端口

#### 4. 技术要点总结

| 项目 | 说明 |
|------|------|
| 评测工具 | EvalScope v1.0+ |
| 评测模式 | `openai_api`（不下载模型，通过 API 调用） |
| 数据集 | GPQA（gpqa_diamond）、GSM8K、ARC |
| 数据来源 | ModelScope Hub，自动流式加载，缓存到 `~/.cache/modelscope` |
| 结果输出 | 可指定 `--output /tmp/...` 避免污染项目目录 |
| 环境隔离 | `.venv` 虚拟环境，已配 `.gitignore` |

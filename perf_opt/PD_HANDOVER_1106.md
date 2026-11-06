# PD 分离任务交接书（2026-11-06，给下一位 Agent）

## 〇、当前阻塞（一句话）

PD 双服务（prefill + decode）无法启动——worker 初始化时 mooncake 引擎崩溃 `std::logic_error: basic_string::_S_construct null not valid`。**修复只需编辑 launcher 一行，然后重启 PD，跑 c32。** 详情见第五节。

## 一、任务目标

在昇腾 910B2C ×16（quant-test 实例）上，给 DeepSeek-V4-Flash（W8A8 量化，280GB）的 PD 分离做**并发扫射 c32 基准**，补齐最后的吞吐天花板数据。已完成 c1/c4/c8/c16（见第三节），只剩 c32。

**最终任务清单**（优先级从高到低）：
1. 修复 PD 启动 → c32 基准 → 得出完整并发曲线
2. 成绩单 `full_optimization_summary.md` 更新 c32 数据
3. logbook 更新 11-06 记录
4. 镜像保存 + 关闭实例（double check STOPPED）
5. 全程邮件汇报 yyc2660109300@outlook.com

## 二、环境全貌（11-06 当前状态）

| 项目 | 值 |
|:-----|:-----|
| 实例 | `quant-test` / workspace `昇腾卡公共空间` |
| 当前镜像 | `glm53-w8a8-env:stopsave-quant-test-b0247-20261104175443`（11-04 用户手动保存，含全部 PD 修复） |
| 当前节点 | `infra-gpu-npu-430.host.shzhisuan.com`（可能在 ticks 之间自动变更） |
| 模型 | `/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-npu16`（280GB） |
| vLLM | 0.23.0 + vllm-ascend；CANN 9.0.0；Python 3.13 |
| Mooncake 引擎 | 自编译 Ascend 版（engine.so + ascend_transport.so），构建树在 `/inspire/sj-ssd3/project/project-public/s26068/mooncake-src/build` |
| 远端 repo | `/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test`（git main, 最新提交 87b9520） |
| 远程连接 | `inspire notebook exec quant-test --workspace 昇腾卡公共空间 '<cmd>'`；如需上传文件用 hex 分块（<1500 字符/块，`bytes.fromhex` + mode=`wb`/`ab` + stdin=`subprocess.DEVNULL`） |

## 三、已完成的成果（全部已保存，不可丢失）

### 3.1 并发扫射数据（11-04，同一镜像，已验证）

| 并发 (P/D) | 请求数 | P prefill p50 | D TTFT p50 | D TTFT p99 | 吞吐 | vs c1 |
|:--|:--|:--|:--|:--|:--|:--|
| c=1 | 16 | 254ms | 394ms | — | 39.5 tok/s | 1× |
| c=4 | 32 | — | 394ms | 655ms | 149.8 tok/s | 3.8× |
| c=8 | 32 | 1235ms | 599ms | 721ms | 274.6 tok/s | 7.0× |
| c=16 | 64 | 2202ms | 617ms | 1017ms | 486.2 tok/s | 12.3× |
| c=32 | 64 | **❌ 未跑** | | | | |

趋势：吞吐随并发线性爬升至 486.2 tok/s（vs 单服 MTP ON c32=781.8 的 62%）。P 是绝对瓶颈（prefill 排队从 254ms 暴涨到 2202ms），D TTFT p50 几乎不退化。c32 可能继续爬升或遇到 P 完全饱和的天花板——这是唯一缺失的数据点。

### 3.2 成绩单 (full_optimization_summary.md)

**本地路径**：`perf_opt/summary/full_optimization_summary.md`

已包含：
- 章节 1-6：MTP 优化、双独立 TP8、基线数据、选择指引（全部最终定稿）
- 章节 7：PD 分离完整实验（两条路线对照、跨场景全表 4K/16K、不对称部署失败记录、并发扫射数据）
- 章节 8：资产索引与数据溯源

**需要补充**：7.3.2 并发表加入 c32 行（c32 完成后执行）

### 3.3 logbook

**本地路径**：`logbook.md`

最新条目：`# 2026-11-04 — PD 攻坚第九回合`（并发扫射 + metrics 分析）

**需要补充**：11-05 和 11-06 的记录（c32 完成后执行）

### 3.4 补丁归档

**本地路径**：`perf_opt/scripts/pd_mooncake_patches/`（7 个补丁 + README 应用顺序与契约）

这些补丁**已在当前 stopsave 镜像中全部应用完毕**，正常情况下无需重跑。如果容器重建或引擎 lib 被覆盖，按 README 重打。

### 3.5 远端关键文件

| 文件 | 用途 |
|:-----|:-----|
| `/tmp/pd_conc_bench.py` | 并发基准客户端（两阶段，P pool + D pool，支持 --p-concurrency/--d-concurrency/--num-requests/--input-chars） |
| `/tmp/pd_orch_bench.py` | 编排基准客户端（单请求模式，含端到端计时和 TPOT 追踪） |
| `/tmp/pd_metrics.py` | metrics 前后差值采集 |
| `/tmp/fix_mooncake_libs.py` | 自动补全 mooncake 目录缺失的 CANN 依赖库 |
| `/tmp/pd_test/_conc_c[1|4|8|16].log` | 11-04 四级并发原始日志 |
| `/tmp/pd_test/_moonFINAL.log` | 最近一次 PD 启动日志（崩溃前） |
| `/inspire/.../perf_opt/scripts/start_pd_mooncake.py` | PD 启动器（当前 git 版本 87b9520，需修一行） |

## 四、成绩单中已修正的已知错误（已审计过，不要再动）

1. **3.1 表**：原先假造了一个"MTP OFF"列（277.8+282.3），实为 c8 MTP=ON 数据。已更正为无双服务 MTP=OFF 实测数据（以单服投影说明 +47.6%）。
2. **GPQA**：原先"vs 官方 72.22%"实为自量化旧版测试值（143/198），已更正为 ModelScope 官方 71.2%。
3. **PD+MTP 比 PD 无 MTP 慢 128ms** 的异常点已配解释块（draft KV 同步与编排开销）。
4. **接受率分点** ~98%/98%/96% 已标注为近似值，以全局 95.79% 为准。

## 五、当前崩溃的根因与一步修复

### 根因

11-05/06 经历多轮反复修改 launcher 的环境变量、重启实例和镜像切换。最终确认的稳定配置需要**同时满足**以下条件：

1. mooncake 目录只留核心 .so 文件（已做：21 个核心文件，34 个 CANN 重复库在 `bak_libs/`）
2. LD_LIBRARY_PATH 末尾追加 mooncake 目录（`$LD_LIBRARY_PATH:/opt/mamba/lib/python3.13/site-packages/mooncake`）——系统 CANN 路径优先，mooncake 兜底
3. 全部 MC_* 环境变量存在（git 版本有）——这些变量每一个都堵住 mooncake C++ 引擎里一个 `getenv()` 返回 NULL 后直接构造 `std::string` 导致的 `basic_string::_S_construct null not valid` 崩溃
4. **删除 `LD_PRELOAD=/tmp/bt_hook2.so`**（10-30 的调试钩子文件已被容器重建清除，预加载无效文件导致所有子进程的 loader 报错）
5. 连接器预加载补丁在岗（`PD fix: preload mooncake` 注释标识）

git 恢复后的 launcher 满足条件 3、条件 4 不满足（LD_PRELOAD 指向不存在的文件）、条件 5 满足（镜像已含补丁）。

### 修复

launcher 文件：`/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/perf_opt/scripts/start_pd_mooncake.py`

**第一步**：删除第 85 行（`export LD_PRELOAD=/tmp/bt_hook2.so`，git 版本里有）

**第二步**：在第 85 行位置（删除后）插入：`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/mamba/lib/python3.13/site-packages/mooncake`

**第三步**：重启 PD：
```bash
pkill -9 -f "VLLM::" 2>/dev/null; pkill -9 -f start_pd_mooncake 2>/dev/null; tmux kill-server 2>/dev/null; sleep 2
cd /inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test
setsid nohup python3 -u perf_opt/scripts/start_pd_mooncake.py --serial > /tmp/pd_test/_moon.log 2>&1 < /dev/null &
```

**第四步**：等双健康（约 5-8 分钟图编译），监控：
```bash
# 每 30 秒检查
curl -s -o /dev/null -w "%{http_code}" -m 4 http://127.0.0.1:8000/health
curl -s -o /dev/null -w "%{http_code}" -m 4 http://127.0.0.1:8001/health
# 如果崩溃立即检查 /tmp/pd_test/decode_v2.log 的 RuntimeError
```

**第五步**：双就绪后跑 c32：
```bash
setsid nohup python3 -u /tmp/pd_conc_bench.py --num-requests 64 --p-concurrency 32 --d-concurrency 32 --input-chars 5400 > /tmp/pd_test/_conc_c32.log 2>&1 < /dev/null &
```

### 验证修复是否生效的检查清单

| 检查项 | 命令 | 期望 |
|:--|:--|:--|
| 引擎加载 | `python3 -c "from mooncake.engine import TransferEngine; e=TransferEngine(); print('OK')"` | OK |
| 引擎+torch 共存 | `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/mamba/.../mooncake ASCEND_RT_VISIBLE_DEVICES=0 python3 -c "import torch,torch_npu; from mooncake.engine import TransferEngine; e=TransferEngine(); print('BOTH OK')"` | BOTH OK |
| mooncake 目录清洁 | `ls /opt/mamba/.../mooncake/*.so* | wc -l` | 21 |
| CANN 重复库已移走 | `ls /opt/mamba/.../mooncake/bak_libs/ | wc -l` | 34 |
| launcher 含 LD_LIBRARY_PATH | `grep "LD_LIBRARY_PATH.*mooncake" .../start_pd_mooncake.py` | 一行 |
| launcher 不含 LD_PRELOAD | `grep "LD_PRELOAD" .../start_pd_mooncake.py` | 零行 |
| launcher 含全部 MC_* 变量 | `grep -c "MC_METADATA_CLUSTER_ID" .../start_pd_mooncake.py` | 1 |
| 连接器预加载在岗 | `grep -c "PD fix: preload" /opt/mamba/.../mooncake_connector.py` | 1 |

### 环境崩溃回滚（如果修不好）

如果上述修复后 PD 仍崩溃，**放弃手修，从 stopsave 镜像重建**：
1. 在 Web UI 把 quant-test 镜像切到 `stopsave-quant-test-b0247-20261104175443`
2. 按本节"修复"步骤执行（LD_PRELOAD 删 + LD_LIBRARY_PATH 加）
3. 该镜像已含全部 6 个连接器补丁，只需修 launcher 这一行

## 六、资产索引（全部路径，供下一位查验）

| 类别 | 路径 |
|:-----|:-----|
| 成绩单 | `perf_opt/summary/full_optimization_summary.md` |
| logbook | `logbook.md` |
| PD 补丁集 | `perf_opt/scripts/pd_mooncake_patches/README.md` |
| PD 问题报告 | `perf_opt/PD_ISSUE_REPORT_1028.md` |
| PD 操作手册 | `perf_opt/PD_HANDOFF_1028.md` |
| 基线数据 | `perf_opt/results/mtp_model/`、`perf_opt/results/verify_1026/` |
| 并发基准客户端 | 远端 `/tmp/pd_conc_bench.py` |
| metrics 采集脚本 | 远端 `/tmp/pd_metrics.py` |
| PD 启动器 | 远端 `.../perf_opt/scripts/start_pd_mooncake.py` |
| 模型 | 远端 `.../models/DeepSeek-V4-Flash-w8a8-npu16` |
| 引擎构建树 | 远端 `/inspire/.../mooncake-src/build` |
| 镜像 | `glm53-w8a8-env:stopsave-quant-test-b0247-20261104175443` |
| git tag | `perf-opt-final` |
| 本地 git 最新提交 | 77870c0 |
| 远端 git 最新提交 | 87b9520 |
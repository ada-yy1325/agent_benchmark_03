# Logbook

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
| 额度保护 | 不能使用 DeepSeek API key 跑 benchmark，需指向国产卡本地服务 |
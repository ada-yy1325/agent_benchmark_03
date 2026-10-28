# 平台环境留证（2026-10-28，供工单）

采集时间：2026-10-28 13:0x（回归发生后）
节点：infra-gpu-npu-416（10-27 下午起三节点 412/416/420 均复现故障）

## 版本信息

| 组件 | 版本 |
|:-----|:-----|
| npu-smi | 25.0.rc1.1 |
| torch_npu | 2.11.0.rc4 |
| CANN 目录 | /usr/local/Ascend/cann-9.0.0 |
| 容器镜像 | glm53-w8a8-env:v2 |
| vLLM | 0.23.0 + vllm-ascend |

注：/usr/local/Ascend/driver/ 下无 version.info（容器内仅 include/lib64）；空闲 NPU 的 HBM-Usage 基线 ~3.4GB/卡（driver 保留）。

## 故障签名

```
RuntimeError: fill_: aclnnInplaceFillScalar failed, error code is 561103
FillAiCore ADD_TO_LAUNCHER_LIST_AICORE failed
RuntimeError: dynamic_quant_run_aclnn ... AclNN_Parameter_Error(EZ1001)
EZ9999 AclNN_Inner_Err
```

- 发生阶段：vLLM worker profile_run
- 触发条件：带图编译的启动必崩；--enforce-eager 普通服务可用；**任何 kv-transfer-config（Mooncake/AscendStore）在 eager 下也崩**（连接器初始化调用坏掉的 fill 算子）
- 时间线：10-27 上午正常（同镜像同脚本），下午起三节点全复现 → 判定为 10-27 下午平台更新引入
- 正常参照：10-26 全流程（双服务 MTP=ON 1588.6 tok/s 实测）在同平台正常完成

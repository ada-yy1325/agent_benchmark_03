#!/bin/bash
export ASCEND_DEVICE_ID=1
msmodelslim quant \
  --model_path /inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash \
  --save_path /inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-self-npu \
  --device npu:1 \
  --model_type DeepSeek-V4-Flash \
  --config_path /opt/mamba/lib/python3.13/site-packages/msmodelslim/lab_practice/deepseek_v4/deepseek_v4_flash_w8a8.yaml \
  --trust_remote_code True 2>&1 | tee /inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/quant_dsv4_npu.log
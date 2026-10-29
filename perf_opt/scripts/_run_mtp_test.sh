#!/bin/bash
# MTP test - wrapper shell script for tmux
set -e
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024 TASK_QUEUE_ENABLE=1 HCCL_OP_EXPANSION_MODE=AIV
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
cd /inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test
exec python3 -B perf_opt/scripts/mtp_test.py 2>&1 | tee /tmp/pd_test/mtp_vllm.log
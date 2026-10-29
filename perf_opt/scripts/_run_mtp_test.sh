#!/bin/bash
# MTP test - run vllm serve with MTP enabled on cards 8-15
set -e
export OMP_PROC_BIND=false OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024 TASK_QUEUE_ENABLE=1 HCCL_OP_EXPANSION_MODE=AIV
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15

MODEL=/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/models/DeepSeek-V4-Flash-w8a8-npu16

exec vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8002 \
  --max-model-len 133120 --max-num-batched-tokens 8192 \
  --served-model-name dsv4_mtp_test --gpu-memory-utilization 0.9 \
  --max-num-seqs 8 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --quantization ascend --block-size 128 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[8,16,24,32,40,48,56,64]}' \
  --speculative-config '{"num_speculative_tokens":1,"method":"mtp"}' \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"enable_dsa_cp":true,"multistream_overlap_shared_expert":true}' \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  2>&1 | tee /tmp/pd_test/mtp_vllm.log
#!/bin/bash
cd /inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test
rm -f eval_full.log eval_gsm8k_base_correct_results.json eval_checkpoint.json
git checkout 552d823 -- eval_gsm8k_base_correct.py
nohup python3 -u eval_gsm8k_base_correct.py --port 8802 > eval_full.log 2>&1 &
echo "PID: $!"
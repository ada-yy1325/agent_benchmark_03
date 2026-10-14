#!/usr/bin/env python3
import json, glob, os

base = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/perf_opt/results/sched/bat_scan"
for f in sorted(glob.glob(os.path.join(base, "bat*.json"))):
    d = json.load(open(f))
    print(f"{os.path.basename(f):30s} TTFT_p50={d['ttft_ms']['p50']:7.0f}ms  TP={d['output_tokens_per_s']:6.1f}  ok={d['successful_requests']}/{d['planned_requests']}")
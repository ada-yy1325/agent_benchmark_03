p = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/perf_opt/scripts/start_pd_mooncake.py"
s = open(p).read()
old = "export HCCL_INTRA_ROCE_ENABLE=1"
new = old + chr(10) + "export ASCEND_BUFFER_POOL=4:8"
assert old in s, "anchor missing"
open(p, "w").write(s.replace(old, new, 1))
print("BUFFER POOL ADDED")

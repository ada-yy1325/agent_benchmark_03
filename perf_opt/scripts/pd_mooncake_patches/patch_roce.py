p = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/perf_opt/scripts/start_pd_mooncake.py"
s = open(p).read()
old = "export MC_TCP_ENABLE_CONNECTION_POOL=0"
new = "export HCCL_INTRA_ROCE_ENABLE=1" + chr(10) + old
assert old in s, "anchor missing"
open(p, "w").write(s.replace(old, new, 1))
print("ROCE ENV ADDED")

p = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/perf_opt/scripts/start_pd_mooncake.py"
s = open(p).read()
old = """    "kv_connector_extra_config": {
        "prefill": {"dp_size": 1, "tp_size": 8},
        "decode": {"dp_size": 1, "tp_size": 8},
    },"""
new = """    "kv_connector_extra_config": {
        "mooncake_protocol": "ascend",
        "prefill": {"dp_size": 1, "tp_size": 8},
        "decode": {"dp_size": 1, "tp_size": 8},
    },"""
assert old in s, "anchor missing"
open(p, "w").write(s.replace(old, new, 1))
print("PROTOCOL SET TO ascend")

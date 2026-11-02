# 1. remove MTP from the mooncake launcher decode side
p1 = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/perf_opt/scripts/start_pd_mooncake.py"
s = open(p1).read()
old = """    if not is_prefill:
        args.extend([
            "--speculative-config",
            '{"num_speculative_tokens": 1, "method": "mtp", "enforce_eager": true}',
        ])"""
new = """    # MTP removed for P/D region symmetry (draft KV caches break the
    # Mooncake asymmetric-TP region count validation on 0.23.0)
    if not is_prefill:
        pass"""
assert old in s, "mtp anchor missing"
open(p1, "w").write(s.replace(old, new, 1))
print("MTP REMOVED")

# 2. add region count log to the connector
p2 = "/opt/mamba/lib/python3.13/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"
s = open(p2).read()
old2 = "        ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)"
new2 = "        logger.info(\"PDCNT registered regions=%d blocks_first=%s\", len(kv_data_ptrs), self.transfer_topo.virtually_split_kv_in_blocks)\n        ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)"
assert old2 in s, "cnt anchor missing"
open(p2, "w").write(s.replace(old2, new2, 1))
print("COUNT LOG ADDED")

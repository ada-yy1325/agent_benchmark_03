import json, glob, sys

files = sorted(glob.glob("/tmp/pd_bench/*.json"))
print(f"Found {len(files)} result files")
print(f"{'case_id':<40} {'ok':>4} {'TTFT p50':>9} {'TTFT p99':>9} {'TPOT p50':>9} {'tok/s':>8}")
print("-" * 85)
for f in files:
    d = json.load(open(f))
    cid = d.get("case_id", "?")
    succ = d.get("successful_requests", "?")
    plan = d.get("planned_requests", "?")
    ttft = d.get("ttft_ms", {})
    tpot = d.get("tpot_ms", {})
    tp = d.get("output_tokens_per_s", "?")
    t50 = ttft.get("p50", -1) if ttft else -1
    t99 = ttft.get("p99", -1) if ttft else -1
    tp50 = tpot.get("p50", -1) if tpot else -1
    print(f"{cid:<40} {succ:>4} {t50:>8.1f}ms {t99:>8.1f}ms {tp50:>8.1f}ms {tp:>8.2f}")

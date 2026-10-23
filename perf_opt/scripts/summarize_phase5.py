#!/usr/bin/env python3
"""Summarize Phase 5.3 — compare all 4 switches against baseline."""
import json, os

BASELINE_PATH = "/tmp/phase5/5_0_baseline.json"
OUT_JSON = "/tmp/phase5/5_3_summary.json"
SWITCHES = [
    "add_cpu_binding_off",
    "add_dsa_cp_off",
    "add_multistream_overlap_off",
    "static_kernel_on",
]
CASES = ["c1_4k", "c8_4k", "c32_4k", "c64_4k"]

# Load baseline
baseline = json.load(open(BASELINE_PATH))
base_data = {}
for pt in baseline["points"]:
    c = pt["concurrency"]
    base_data[c] = {
        "ttft_ms": pt["ttft_ms"]["p50"],
        "tok_s": pt["output_tokens_per_s"],
    }

print("Baseline (5.0):")
for c in [1, 8, 32, 64]:
    d = base_data[c]
    print(f"  c{c}: TTFT={d['ttft_ms']:.0f}ms tok/s={d['tok_s']:.1f}")
print()

summary = {"baseline": {}, "switches": {}}
for c in [1, 8, 32, 64]:
    summary["baseline"][f"c{c}"] = base_data[c]

for switch in SWITCHES:
    d = f"/tmp/phase5/{switch}"
    print(f"{switch}:")
    sw_data = {}
    for case in CASES:
        fpath = f"{d}/{switch}_{case}.json"
        if os.path.isfile(fpath):
            j = json.load(open(fpath))
            ttft = j["ttft_ms"]["p50"]
            tok = j["output_tokens_per_s"]
            cn = j["concurrency"]
            baseline_ttft = base_data[cn]["ttft_ms"]
            baseline_tok = base_data[cn]["tok_s"]
            d_ttft = (ttft - baseline_ttft) / baseline_ttft * 100
            d_tok = (tok - baseline_tok) / baseline_tok * 100
            print(f"  c{cn}: TTFT={ttft:.0f}ms ({d_ttft:+.1f}%) tok/s={tok:.1f} ({d_tok:+.1f}%)")
            sw_data[f"c{cn}"] = {
                "ttft_ms": ttft,
                "tok_s": tok,
                "delta_ttft_pct": round(d_ttft, 1),
                "delta_tok_pct": round(d_tok, 1),
            }
        else:
            print(f"  {case}: MISSING")
    summary["switches"][switch] = sw_data
    print()

# Write summary JSON
json.dump(summary, open(OUT_JSON, "w"), indent=2)
print(f"Saved to {OUT_JSON}")
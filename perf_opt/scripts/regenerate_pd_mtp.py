#!/usr/bin/env python3
"""Regenerate PD MTP launch scripts (with --speculative-config)."""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from perf_opt.scripts.start_pd_mooncake import build_args, generate_shell_script

TMP = "/tmp/pd_test"
os.makedirs(TMP, exist_ok=True)

# Generate args with MTP=True for both P and D
for pf, port, lb in [(True, 8000, "prefill"), (False, 8001, "decode")]:
    a = build_args(prefill=pf, port=port, mtp=True)
    out = f"{TMP}/pd_{lb}_mtp_args.json"
    with open(out, "w") as f:
        json.dump(a, f)
    print(f"Generated {out} ({len(a)} args, MTP ON)")

# Generate full launch scripts with all env vars
for role, devs, log, af in [
    ("pd_decode", "8,9,10,11,12,13,14,15", "decode_mtp.log",
     f"{TMP}/pd_decode_mtp_args.json"),
    ("pd_prefill", "0,1,2,3,4,5,6,7", "prefill_mtp.log",
     f"{TMP}/pd_prefill_mtp_args.json"),
]:
    s = generate_shell_script(role, devs, af, f"{TMP}/{log}")
    out = f"{TMP}/_{role}_mtp_launch.sh"
    with open(out, "w") as f:
        f.write(s)
    os.chmod(out, 0o755)
    print(f"Created {out}")

print("DONE - MTP launch scripts ready")
print(f"Run: bash {TMP}/_pd_prefill_mtp_launch.sh & sleep 10 && bash {TMP}/_pd_decode_mtp_launch.sh &")
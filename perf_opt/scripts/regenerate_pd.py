#!/usr/bin/env python3
"""Regenerate PD launch scripts from start_pd_mooncake.py build_args()."""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perf_opt.scripts.start_pd_mooncake import build_args, generate_shell_script

TMP = "/tmp/pd_test"
os.makedirs(TMP, exist_ok=True)

for pf, port, lb in [(True, 8000, "pd_prefill"), (False, 8001, "pd_decode")]:
    a = build_args(prefill=pf, port=port)
    with open(f"{TMP}/pd_{lb}_args.json", "w") as f:
        json.dump(a, f)

for role, devs, log, af in [
    ("pd_decode", "8,9,10,11,12,13,14,15", "decode_v2.log", "pd_decode_args.json"),
    ("pd_prefill", "0,1,2,3,4,5,6,7", "prefill_v2.log", "pd_prefill_args.json"),
]:
    s = generate_shell_script(role, devs, f"{TMP}/{af}", f"{TMP}/{log}")
    with open(f"{TMP}/_{role}_launch.sh", "w") as f:
        f.write(s)
    os.chmod(f"{TMP}/_{role}_launch.sh", 0o755)

print("REGENERATED OK")
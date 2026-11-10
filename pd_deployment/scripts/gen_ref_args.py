#!/usr/bin/env python3
"""Generate known-working .ref.json arg files for pd_deployment/."""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from perf_opt.scripts.start_pd_mooncake import build_args

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# No-MTP reference args
for pf, port, lb in [(True, 8000, "prefill"), (False, 8001, "decode")]:
    a = build_args(prefill=pf, port=port, mtp=False)
    path = os.path.join(ROOT, "pd_deployment", "non_mtp", f"pd_{lb}_args.ref.json")
    with open(path, "w") as f:
        json.dump(a, f, indent=2)
    print(f"non_mtp/pd_{lb}_args.ref.json: {len(a)} args")

# MTP reference args
for pf, port, lb in [(True, 8000, "prefill"), (False, 8001, "decode")]:
    a = build_args(prefill=pf, port=port, mtp=True)
    path = os.path.join(ROOT, "pd_deployment", "mtp", f"pd_{lb}_mtp_args.ref.json")
    with open(path, "w") as f:
        json.dump(a, f, indent=2)
    print(f"mtp/pd_{lb}_mtp_args.ref.json: {len(a)} args")

print("DONE")
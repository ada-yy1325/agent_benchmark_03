#!/usr/bin/env python3
"""Add MTP to P/D args and generate launch scripts."""
import json, os, shlex

TMP = "/tmp/pd_test"
os.makedirs(TMP, exist_ok=True)

# Load existing args
pf = json.load(open(f"{TMP}/pd_prefill_args.json"))
dc = json.load(open(f"{TMP}/pd_decode_args.json"))

# Add MTP speculative config to both P and D
mtp_cfg = '{"method":"mtp","num_speculative_tokens":1}'
for name, args in [("prefill", pf), ("decode", dc)]:
    if "--speculative-config" not in args:
        idx = args.index("--compilation-config")
        args.insert(idx, "--speculative-config")
        args.insert(idx + 1, mtp_cfg)
    print(f"[{name}] added MTP, total args={len(args)}")

# Save
json.dump(pf, open(f"{TMP}/pd_prefill_mtp_args.json", "w"), indent=2)
json.dump(dc, open(f"{TMP}/pd_decode_mtp_args.json", "w"), indent=2)
print("MTP args saved")

# Generate tmux launch script
BASE = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test"
VLLM = "/opt/mamba/bin/vllm"

for role, devices, args_file, tmux_name in [
    ("prefill", "0,1,2,3,4,5,6,7", "pd_prefill_mtp_args.json", "pd_prefill_mtp"),
    ("decode", "8,9,10,11,12,13,14,15", "pd_decode_mtp_args.json", "pd_decode_mtp"),
]:
    with open(f"{TMP}/{args_file}") as f:
        args = json.load(f)

    cmd = [VLLM, "serve"]
    for a in args:
        if a.startswith("-"):
            cmd.append(a)
        elif a.startswith("{"):
            cmd.append(a)
        else:
            cmd.append(a)

    quoted_cmd = " ".join(shlex.quote(a) for a in cmd)

    script = f"""#!/bin/bash
# PD MTP {role} - tmux session: {tmux_name}
export ASCEND_RT_VISIBLE_DEVICES={devices}
export HCCL_INTRA_ROCE_ENABLE=1
export ASCEND_BUFFER_POOL=4:8
export MC_FORCE_TCP=0
cd {BASE}

echo "Starting {role} with MTP in tmux session {tmux_name}..."
{quoted_cmd} 2>&1 | tee /tmp/pd_test/{role}_mtp.log
"""
    out_path = f"{TMP}/_{role}_mtp.sh"
    with open(out_path, "w") as f:
        f.write(script)
    os.chmod(out_path, 0o755)
    print(f"Created {out_path}")

# Also create a combined start script with tmux
tmscript = f"""#!/bin/bash
# Kill existing P/D sessions
tmux kill-session -t pd_prefill_mtp 2>/dev/null || true
tmux kill-session -t pd_prefill 2>/dev/null || true
tmux kill-session -t pd_decode_mtp 2>/dev/null || true
tmux kill-session -t pd_decode 2>/dev/null || true
sleep 3

# Start P (prefill/producer) in tmux
echo "Starting P (prefill) with MTP..."
tmux new-session -d -s pd_prefill_mtp \\
  "export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
   export HCCL_INTRA_ROCE_ENABLE=1 \\
   export ASCEND_BUFFER_POOL=4:8 \\
   export MC_FORCE_TCP=0 \\
   cd {BASE} && bash {TMP}/_prefill_mtp.sh"

sleep 5

# Start D (decode/consumer) in tmux  
echo "Starting D (decode) with MTP..."
tmux new-session -d -s pd_decode_mtp \\
  "export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \\
   export HCCL_INTRA_ROCE_ENABLE=1 \\
   export ASCEND_BUFFER_POOL=4:8 \\
   export MC_FORCE_TCP=0 \\
   cd {BASE} && bash {TMP}/_decode_mtp.sh"

echo "PD MTP tmux sessions started"
echo "Check: tmux ls"
echo "P logs: tail -f /tmp/pd_test/prefill_mtp.log"
echo "D logs: tail -f /tmp/pd_test/decode_mtp.log"
"""

tms_path = f"{TMP}/start_pd_mtp.sh"
with open(tms_path, "w") as f:
    f.write(tmscript)
os.chmod(tms_path, 0o755)
print(f"Created tmux starter: {tms_path}")
print("DONE - Run: bash /tmp/pd_test/start_pd_mtp.sh")

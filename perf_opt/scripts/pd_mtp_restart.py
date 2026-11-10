# Add MTP to P/D args and generate launch scripts
import json, os

TMP = "/tmp/pd_test"

# Load existing args
pf = json.load(open(f"{TMP}/pd_prefill_args.json"))
dc = json.load(open(f"{TMP}/pd_decode_args.json"))

# Add MTP speculative config
mtp_cfg = '{"method":"mtp","num_speculative_tokens":1}'

for args in [pf, dc]:
    if "--speculative-config" not in args:
        # Insert before --compilation-config
        idx = args.index("--compilation-config")
        args.insert(idx, "--speculative-config")
        args.insert(idx + 1, mtp_cfg)

# Save MTP args
json.dump(pf, open(f"{TMP}/pd_prefill_mtp_args.json", "w"))
json.dump(dc, open(f"{TMP}/pd_decode_mtp_args.json", "w"))

# Generate launch scripts for MTP
BASE = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test"
VLLM = "/opt/mamba/bin/vllm"

for role, devices, args_file, logfile in [
    ("pd_prefill", "0,1,2,3,4,5,6,7", "pd_prefill_mtp_args.json", "prefill_mtp.log"),
    ("pd_decode", "8,9,10,11,12,13,14,15", "pd_decode_mtp_args.json", "decode_mtp.log"),
]:
    with open(f"{TMP}/{args_file}") as f:
        args = json.load(f)

    lines = [f"#!/bin/bash", f"# PD MTP {role}"]
    lines.append(f"export ASCEND_RT_VISIBLE_DEVICES={devices}")
    lines.append(f"export HCCL_INTRA_ROCE_ENABLE=1")
    lines.append(f"export ASCEND_BUFFER_POOL=4:8")
    lines.append(f"export MC_FORCE_TCP=0")
    lines.append(f"cd {BASE}")
    
    # Build command line
    cmd_parts = [VLLM, "serve"] + [
        f"'{a}'" if (" " in a or "{" in a or ":" in a) and not (a.startswith("'") or a.startswith('"'))
        else a'
        for a in args
    ]
    cmd = " \\\n    ".join(cmd_parts)
    lines.append(f"echo 'Starting {role} with MTP...'")
    lines.append(f"{cmd} > /tmp/pd_test/{logfile} 2>&1")
    
    script = "\n".join(lines) + "\n"
    out_path = f"{TMP}/_{role}_mtp_launch.sh"
    with open(out_path, "w") as f:
        f.write(script)
    os.chmod(out_path, 0o755)
    print(f"Created {out_path}")

print("DONE - MTP launch scripts ready")

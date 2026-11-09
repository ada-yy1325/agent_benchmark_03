#!/usr/bin/env python3
"""
Patch ascend_direct_transport.cpp to tolerate error 103900 (HcclRegisterGlobalMem overlap).
Then rebuild ascend_transport.so and install it.

Error 103900 occurs when TP=8 vLLM workers (separate processes) each try to register
the same NPU KV cache memory via HCCL. The first worker succeeds, subsequent workers get
103900 ("already registered by another process"). This patch treats 103900 as success
rather than failure.

Apply: python3 /tmp/patch_adxl_103900.py
"""

import os, subprocess, shutil, sys

MOONCAKE_SRC = "/inspire/sj-ssd3/project/project-public/s26068/mooncake-src"
MOONCAKE_LIB = "/opt/mamba/lib/python3.13/site-packages/mooncake"

# --- Step 1: Locate the source file ---
# There are TWO copies. The one actually compiled into ascend_transport.so
# is under src/transport/ascend_transport/ascend_direct_transport/.
# The TENT copy is newer but not compiled into the existing .so.
candidates = []
for root, dirs, files in os.walk(os.path.join(MOONCAKE_SRC, "mooncake-transfer-engine")):
    for f in files:
        if f == "ascend_direct_transport.cpp":
            candidates.append(os.path.join(root, f))

if not candidates:
    print("ERROR: ascend_direct_transport.cpp not found")
    sys.exit(1)

# Prefer the non-TENT copy (the one actually compiled)
src_file = None
for c in candidates:
    if "src/transport/ascend_transport" in c:
        src_file = c
        break
if not src_file:
    src_file = candidates[0]
print(f"[patch] Found source: {src_file}")

# --- Step 2: Read and patch ---
with open(src_file) as f:
    src = f.read()

# Target: the registerLocalMemory function's registerMem return check
# Old: if (register_ret == 0) { return 0; }
# New: if (register_ret == 0 || register_ret == 103900) { return 0; }
old = "if (register_ret == 0) {\n        return 0;\n    }"
new = (
    "if (register_ret == 0 || register_ret == 103900) {\n"
    "        // 103900 = HcclRegisterGlobalMem overlap: another worker already\n"
    "        // registered this buffer. Treat as success in multi-process TP setup.\n"
    "        return 0;\n"
    "    }"
)

if old not in src:
    print(f"[patch] ERROR: anchor string not found in {src_file}")
    print(f"[patch] Searching for register_ret pattern...")
    if "register_ret == 0" in src:
        print("[patch] 'register_ret == 0' found but context differs. Searching...")
        idx = src.index("register_ret == 0")
        print(f"[patch] Context at offset {idx}:")
        print(repr(src[idx-20:idx+50]))
    sys.exit(1)

src = src.replace(old, new, 1)
with open(src_file, "w") as f:
    f.write(src)
print("[patch] ERROR 103900 TOLERANCE PATCHED IN ascend_direct_transport.cpp")

# --- Step 3: Also need to ensure MC_FORCE_TCP is NOT set ---
# Check and report
for env_file in [
    "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/perf_opt/scripts/start_pd_mooncake.py",
]:
    if os.path.exists(env_file):
        with open(env_file) as f:
            content = f.read()
        if "MC_FORCE_TCP" in content:
            lines = [l for l in content.split("\n") if "MC_FORCE_TCP" in l]
            print(f"[patch] MC_FORCE_TCP found in {os.path.basename(env_file)}:")
            for l in lines:
                print(f"  {l.strip()}")
        else:
            print(f"[patch] MC_FORCE_TCP NOT found in {os.path.basename(env_file)} ✓")

# --- Step 4: Incremental build ---
build_dir = os.path.join(MOONCAKE_SRC, "build")
if not os.path.exists(os.path.join(build_dir, "Makefile")):
    print("[build] ERROR: build/Makefile not found")
    sys.exit(1)

print("[build] Starting incremental cmake build for ascend_transport...")
result = subprocess.run(
    ["make", "-j4", "ascend_transport"],
    cwd=build_dir,
    capture_output=True, text=True, timeout=600,
)
print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
if result.returncode != 0:
    print(result.stderr[-1000:])
    print("[build] ERROR: build failed")
    sys.exit(1)
print("[build] BUILD SUCCESS")

# --- Step 5: Find and install the new .so ---
for root, dirs, files in os.walk(build_dir):
    for f in files:
        if f == "ascend_transport.so":
            new_so = os.path.join(root, f)
            print(f"[build] New .so: {new_so}")

            # Backup old
            old_so = os.path.join(MOONCAKE_LIB, "ascend_transport.so")
            bak = old_so + ".bak103900"
            if not os.path.exists(bak):
                shutil.copy2(old_so, bak)
                print(f"[install] Backed up: {bak}")

            # Install new
            shutil.copy2(new_so, old_so)
            print(f"[install] Installed new ascend_transport.so → {old_so}")

            # Verify
            with open(new_so, "rb") as a, open(old_so, "rb") as b:
                if a.read() == b.read():
                    print("[install] VERIFIED: binary identical ✓")
                else:
                    print("[install] WARNING: binaries differ!")
            break

print("\n[patch] DONE — ascend_direct_transport.cpp patched + ascend_transport.so rebuilt")
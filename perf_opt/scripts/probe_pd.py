#!/usr/bin/env python3
"""Probe vLLM environment for PD separation (disaggregated prefill) support.

Checks:
  1. vLLM version
  2. CLI args related to disaggregation / KV transfer
  3. Python modules for kv_transfer
  4. Whether --disaggregation-mode is recognized
"""
import os
import sys
import importlib

print("=" * 60)
print("PD Separation Probe")
print("=" * 60)

# 1. Versions
print("\n--- 1. Versions ---")
try:
    import vllm
    print(f"vllm version: {vllm.__version__}")
    print(f"vllm commit: {getattr(vllm, '__commit__', 'N/A')}")
except Exception as e:
    print(f"ERROR importing vllm: {e}")

try:
    import vllm_ascend
    print(f"vllm_ascend version: {getattr(vllm_ascend, '__version__', 'N/A')}")
except Exception as e:
    print(f"vllm_ascend: {e}")

# 2. Check for kv_transfer module
print("\n--- 2. KV Transfer modules ---")
for mod_name in [
    "vllm.distributed.kv_transfer",
    "vllm.distributed.kv_transfer.kv_connector",
    "vllm.distributed.kv_transfer.kv_transfer_state",
    "vllm.distributed.kv_transfer.kv_pipe",
    "vllm.distributed.kv_transfer.lookup_buffer",
    "vllm.engine.kv_transfer",
]:
    try:
        mod = importlib.import_module(mod_name)
        print(f"  ✅ {mod_name} — {getattr(mod, '__file__', 'exists')}")
    except ImportError:
        print(f"  ❌ {mod_name} — NOT FOUND")

# 3. Check CLI via FlexibleArgumentParser
print("\n--- 3. CLI argument probe ---")
try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.openai.cli_args import make_arg_parser

    p = FlexibleArgumentParser(description="probe")
    p = make_arg_parser(p)

    # Collect all registered argument dest names
    all_dests = set()
    for action in p._actions:
        if action.dest and action.dest != "help":
            all_dests.add(action.dest)

    # Look for disaggregation / KV transfer related args
    pd_keywords = [
        "disaggregat", "disagg", "scale_out", "scale-out", "enable_scale_out",
        "kv_transfer", "kv_connector", "kv_ip", "kv_port",
        "kv_load", "kv_store", "tokens_only",
    ]
    found_args = []
    for action in p._actions:
        dest = action.dest
        option_str = ", ".join(action.option_strings) if action.option_strings else ""
        full = f"{option_str} -> {dest}"
        for kw in pd_keywords:
            if kw in full.lower():
                found_args.append(f"  🔍 {full}  (help: {action.help[:100] if action.help else 'N/A'})")
                break
    if found_args:
        print("PD-related args found:")
        for a in found_args:
            print(a)
    else:
        print("❌ NO PD-related CLI arguments found in make_arg_parser")
except Exception as e:
    print(f"ERROR probing CLI: {e}")

# 4. Check if --disaggregation-mode is recognized by running --help
print("\n--- 4. Raw --help grep for disaggregation / kv ---")
import subprocess
try:
    result = subprocess.run(
        ["python3", "-m", "vllm.entrypoints.openai.api_server", "--help"],
        capture_output=True, text=True, timeout=30
    )
    help_text = result.stdout + result.stderr
    for keyword in ["disaggregat", "scale-out", "scale_out", "kv-transfer", "kv_transfer",
                     "kv-connector", "kv_connector", "kv-ip", "kv_ip",
                     "tokens-only", "tokens_only", "prefill"]:
        for line in help_text.split("\n"):
            if keyword.lower() in line.lower():
                print(f"  🔍 [{keyword}]: {line.strip()[:150]}")
except Exception as e:
    print(f"ERROR running --help: {e}")

# 5. Check vllm.config for KVTransferConfig
print("\n--- 5. Config classes ---")
try:
    from vllm.config import VllmConfig
    # Check what configs exist
    import vllm.config as cfg_mod
    for name in dir(cfg_mod):
        if "kv" in name.lower() or "disagg" in name.lower() or "scale" in name.lower():
            print(f"  🔍 vllm.config.{name}")
except Exception as e:
    print(f"ERROR: {e}")

# 6. Try direct: does EngineArgs have disaggregation_mode?
print("\n--- 6. EngineArgs attribute check ---")
try:
    from vllm.engine.arg_utils import EngineArgs
    fields = EngineArgs.__dataclass_fields__ if hasattr(EngineArgs, '__dataclass_fields__') else {}
    for name in sorted(fields.keys()):
        if any(kw in name.lower() for kw in ["disagg", "kv_transfer", "scale_out", "tokens_only"]):
            print(f"  🔍 EngineArgs.{name}: {fields[name]}")
    if not any(kw in name.lower() for name in fields.keys()
               for kw in ["disagg", "kv_transfer", "scale_out"]):
        print("  ❌ No PD-related fields in EngineArgs")
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("Probe complete")
print("=" * 60)
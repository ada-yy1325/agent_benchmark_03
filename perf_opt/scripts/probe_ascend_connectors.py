#!/usr/bin/env python3
"""Probe available KV connectors and their configuration on Ascend NPU."""
import importlib

print("=== Checking KV connector factory ===")
try:
    from vllm.distributed.kv_transfer.kv_connector.factory import (
        get_kv_connector_class,
        KVConnectorFactory,
    )
    # Try to list available connectors
    if hasattr(KVConnectorFactory, '_registry'):
        print("Registry:", list(KVConnectorFactory._registry.keys()))
    if hasattr(KVConnectorFactory, 'available_connectors'):
        print("Available:", KVConnectorFactory.available_connectors())
except Exception as e:
    print(f"Factory check failed: {e}")

print("\n=== Checking vllm_ascend connectors ===")
try:
    import vllm_ascend.distributed.kv_transfer.ascend_multi_connector as amc
    for name in dir(amc):
        if 'onnect' in name.lower():
            print(f"  ascend_multi_connector.{name}")
except Exception as e:
    print(f"ascend_multi_connector: {e}")

try:
    import vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector as mc
    for name in dir(mc):
        if 'onnect' in name.lower():
            print(f"  mooncake_connector.{name}")
    # Check doc or class for connector name
    for name in dir(mc):
        obj = getattr(mc, name)
        if hasattr(obj, 'connector_name'):
            print(f"  {name}.connector_name = {obj.connector_name}")
except Exception as e:
    print(f"mooncake_connector: {e}")

print("\n=== Checking all vllm_ascend.distributed.kv_transfer files ===")
import vllm_ascend, os, inspect
base = os.path.dirname(vllm_ascend.__file__)
kv_dir = os.path.join(base, 'distributed', 'kv_transfer')
for root, dirs, files in os.walk(kv_dir):
    for f in sorted(files):
        if f.endswith('.py') and not f.startswith('_'):
            full = os.path.join(root, f)
            rel = full[len(kv_dir)+1:-3].replace('/', '.')
            mod_path = 'vllm_ascend.distributed.kv_transfer.' + rel
            try:
                mod = importlib.import_module(mod_path)
                for name in dir(mod):
                    if 'onnect' in name.lower() and not name.startswith('_'):
                        print(f"  {mod_path}.{name}")
            except Exception as e:
                if 'No module' not in str(e):
                    print(f"  {mod_path}: ERROR {e}")

print("\nDone")
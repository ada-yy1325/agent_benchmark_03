#!/usr/bin/env python3
"""Quick probe: list KV connector files and available classes."""
import os, importlib, inspect

base = '/opt/mamba/lib/python3.13/site-packages/vllm/distributed/kv_transfer/kv_connector'
print("=== KV connector files ===")
for root, dirs, files in os.walk(base):
    for f in sorted(files):
        if f.endswith('.py'):
            full = os.path.join(root, f)
            print(full.replace(base + '/', ''))

print("\n=== Available Connector classes ===")
for root, dirs, files in os.walk(base):
    for f in sorted(files):
        if f.endswith('.py') and not f.startswith('_'):
            rel = os.path.join(root, f)[len(base)+1:-3].replace('/', '.')
            mod_path = 'vllm.distributed.kv_transfer.kv_connector.' + rel
            try:
                mod = importlib.import_module(mod_path)
                for name, obj in inspect.getmembers(mod, inspect.isclass):
                    if 'onnect' in name.lower():
                        print(f'  {mod_path}.{name}')
            except Exception as e:
                print(f'  {mod_path}: ERROR {e}')

print("\nDone")
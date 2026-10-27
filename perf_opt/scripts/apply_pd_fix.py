#!/usr/bin/env python3
"""
Apply PD fix to AscendStoreConnector on remote NPU machine.
- Skip LookupKeyServer creation for kv_consumer
- Config already supports consumer_is_to_load via kv_connector_extra_config

Usage:
  python3 apply_pd_fix.py
"""
import os
import re

CONNECTOR_PATH = "/opt/mamba/lib/python3.13/site-packages/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py"

def main():
    with open(CONNECTOR_PATH, 'r') as f:
        content = f.read()
    
    # Backup
    backup = content
    
    # Fix 1: Skip LookupKeyServer for consumer
    # Old: if vllm_config.parallel_config.rank == 0:
    # Old:     self.lookup_server = LookupKeyServer(...)
    #
    # New: Only create server if not consumer
    old_pattern = r'(if vllm_config\.parallel_config\.rank == 0:\s+'
    old_pattern += r'self\.lookup_server = LookupKeyServer\(self\.connector_worker, vllm_config, self\.use_layerwise\))'
    
    new_code = (
        '# PD fix: consumer does not need its own LookupKeyServer (queries producer\'s)\n'
        '            is_consumer = self.kv_role == "kv_consumer"\n'
        '            if vllm_config.parallel_config.rank == 0 and not is_consumer:\n'
        '                self.lookup_server = LookupKeyServer(self.connector_worker, vllm_config, self.use_layerwise)'
    )
    
    content = re.sub(old_pattern, new_code, content)
    
    if content == backup:
        print("⚠ No changes made (pattern not found). Checking file...")
        # Show the relevant section
        lines = backup.split('\n')
        for i, line in enumerate(lines):
            if 'lookup_server' in line or 'rank == 0' in line:
                print(f"  L{i+1}: {line}")
    else:
        with open(CONNECTOR_PATH, 'w') as f:
            f.write(content)
        print("✅ PD fix applied to AscendStoreConnector!")
        print("  - Consumer will NOT create LookupKeyServer")
        print("  - Consumer's LookupKeyClient connects to producer's server")
    
if __name__ == "__main__":
    main()
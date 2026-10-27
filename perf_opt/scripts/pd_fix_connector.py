#!/usr/bin/env python3
"""Apply PD fix to AscendStoreConnector: skip LookupKeyServer for consumer."""
import os, shutil

PATH = "/opt/mamba/lib/python3.13/site-packages/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py"

def main():
    # Read
    with open(PATH) as f:
        lines = f.readlines()

    # Backup
    shutil.copy2(PATH, PATH + ".pd_fix_bak")
    print(f"Backup: {PATH}.pd_fix_bak")

    # Find the target block
    old_block = [
        "            if vllm_config.parallel_config.rank == 0:\n",
        "                self.lookup_server = LookupKeyServer(self.connector_worker, vllm_config, self.use_layerwise)\n",
    ]
    new_block = [
        "            # PD fix: kv_consumer queries producer's LookupKeyServer via shared socket\n",
        "            is_consumer = (self.kv_role == \"kv_consumer\")\n",
        "            if vllm_config.parallel_config.rank == 0 and not is_consumer:\n",
        "                self.lookup_server = LookupKeyServer(self.connector_worker, vllm_config, self.use_layerwise)\n",
    ]

    # Replace
    content = "".join(lines)
    old_text = "".join(old_block)
    new_text = "".join(new_block)

    if old_text not in content:
        print("❌ Target pattern not found! Content around lookup_server:")
        for i, line in enumerate(lines):
            if "lookup_server" in line or "rank == 0" in line:
                print(f"  L{i+1}: {line.rstrip()}")
        return 1

    content = content.replace(old_text, new_text)
    with open(PATH, "w") as f:
        f.write(content)

    print("✅ PD fix applied!")
    print("  - kv_consumer instances will NOT create LookupKeyServer")
    print("  - They will connect to kv_producer's LookupKeyServer (same port)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Bypass hybrid KV cache check for layerwise mode in AscendStore."""
import os, sys, shutil

TARGETS = {
    "pool_scheduler": "/opt/mamba/lib/python3.13/site-packages/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py",
    "pool_worker": "/opt/mamba/lib/python3.13/site-packages/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py",
}

PATTERNS = {
    "pool_scheduler": (
        'if self.use_layerwise and len(self.kv_cache_group_ids) > 1:\n'
        '            raise NotImplementedError("AscendStore layerwise mode does not yet support hybrid KV cache groups.")'
    ),
    "pool_worker": (
        'if self.use_layerwise and self.num_kv_cache_groups > 1:\n'
        '            raise NotImplementedError("AscendStore layerwise mode does not yet support hybrid KV cache groups.")'
    ),
}

REPLACEMENTS = {
    "pool_scheduler": (
        'if self.use_layerwise and len(self.kv_cache_group_ids) > 1:\n'
        '            import logging; logging.getLogger(__name__).warning("bypass: layerwise+hybrid (experimental)")'
    ),
    "pool_worker": (
        'if self.use_layerwise and self.num_kv_cache_groups > 1:\n'
        '            import logging; logging.getLogger(__name__).warning("bypass: layerwise+hybrid (experimental)")'
    ),
}

for name, target_path in TARGETS.items():
    if not os.path.exists(target_path):
        print(f"ERROR: {name} not found at {target_path}")
        sys.exit(1)

    with open(target_path, 'r') as f:
        content = f.read()

    old = PATTERNS[name]
    new = REPLACEMENTS[name]

    if old in content:
        bak = target_path + '.pd_hybrid_bak'
        if not os.path.exists(bak):
            shutil.copy2(target_path, bak)
            print(f"Backup: {bak}")
        content = content.replace(old, new)
        with open(target_path, 'w') as f:
            f.write(content)
        print(f"Patched: {name}")
    else:
        print(f"Already patched: {name}")

print("Done.")
#!/usr/bin/env python3
"""PD surgical fix (file-level, idempotent): full-prompt block_hashes for pool lookup.

Root cause (2026-10-27): vLLM V1 fills request.block_hashes incrementally; at
the first scheduling step only the first chunk's hashes exist, so AscendStore's
pool lookup can only match the first ~512 tokens and the consumer recomputes
the rest locally.

Fix: insert a small block into KVPoolScheduler.get_num_new_matched_tokens
(pool_scheduler.py) that extends block_hashes to cover the whole prompt by
continuing vLLM's own chained block hash (hash_block_tokens) over the
request's prompt_token_ids. Same environment + same hash function => keys
match the producer's saved entries exactly.

Usage: python3 perf_opt/scripts/pd_full_hash_fix.py   # idempotent; re-run safe
"""
import os
import re
import sys

PKG = "/opt/mamba/lib/python3.13/site-packages"
TARGET = f"{PKG}/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py"

FIX_MARK = "# === PD full-hash fix (2026-10-27)"

INSERTION = FIX_MARK + """
        try:
            gs = int(getattr(self, "group_block_size", 128) or 128)
            hbs = int(getattr(self, "hash_block_size", 128) or 128)
            needed = (token_len + gs - 1) // gs
            if request.block_hashes and len(request.block_hashes) < needed:
                tokens = request.prompt_token_ids
                hashes = list(request.block_hashes)
                for g in range(len(hashes), needed):
                    start = g * gs
                    end = min(start + gs, token_len)
                    parent = hashes[-1]
                    for b in range(start, end, hbs):
                        be = min(b + hbs, end)
                        parent = hash_block_tokens(parent, tuple(tokens[b:be]), None)
                    hashes.append(parent)
                request.block_hashes = hashes
        except Exception:
            pass
        # === end PD full-hash fix ===
"""


def main():
    if not os.path.exists(TARGET):
        print(f"[pd-full-hash] ERROR: {TARGET} not found")
        return 1
    src = open(TARGET).read()
    if FIX_MARK in src:
        print("[pd-full-hash] already patched")
        return 0

    # 1. ensure hash_block_tokens import
    imp = "from vllm.v1.core.kv_cache_utils import hash_block_tokens"
    if imp not in src:
        # insert after the last existing "from vllm" import
        imports = [m.end() for m in re.finditer(r"^from vllm[^\n]*\n", src, re.M)]
        pos = max(imports) if imports else 0
        src = src[:pos] + imp + "\n" + src[pos:]
        print("[pd-full-hash] added hash_block_tokens import")

    # 2. insert the extension right before the client.lookup call
    anchor = "        num_external_hit_tokens = self.client.lookup("
    if anchor not in src:
        print("[pd-full-hash] ERROR: lookup anchor not found")
        return 1
    src = src.replace(anchor, INSERTION + anchor, 1)

    # 3. backup + write
    if not os.path.exists(TARGET + ".pd_bak"):
        os.system(f"cp {TARGET} {TARGET}.pd_bak")
    open(TARGET, "w").write(src)
    print("[pd-full-hash] patch written (backup: pool_scheduler.py.pd_bak)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

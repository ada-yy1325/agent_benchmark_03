#!/usr/bin/env python3
"""Canonical MooncakeConnector patches for DSV4 hybrid KV on vllm 0.23.0 (2026-10-30).

Apply to /opt/mamba/lib/python3.13/site-packages/vllm/distributed/kv_transfer/
kv_connector/v1/mooncake/mooncake_connector.py (official v0.23.0 pristine file).

Patches:
1. PRELOAD: dlopen mooncake Ascend-build deps by absolute path before the
   `from mooncake.engine import TransferEngine` import (no LD_LIBRARY_PATH
   pollution - any CANN path in LD breaks torch_npu's load check).
2. FLATTEN: hybrid KV gives per-layer LISTS of tensors (one per group).
3. GLOBAL DEDUP: the Mooncake engine rejects overlapped memory regions and
   checks only consecutive pairs after sorting by address. Hybrid KV views
   alias shared allocations ACROSS groups and k/v dict keys, so dedupe
   globally with the engine's exact rule.
4. PDX1-4 probes: INFO logs on the receive/send chain (for debugging the
   current transfer hang). REMOVE after the transfer is fixed.
"""
import sys

P = "/opt/mamba/lib/python3.13/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"

PRELOAD = """    # PD fix: preload mooncake Ascend build deps by absolute path (no LD_LIBRARY_PATH pollution)
    import ctypes
    for _dep in (
        "/opt/mamba/lib/libyaml-cpp.so.0.8",
        "/opt/mamba/lib/python3.13/site-packages/mooncake/libmooncake_common.so",
        "/opt/mamba/lib/python3.13/site-packages/mooncake/libtransfer_engine.so",
        "/opt/mamba/lib/python3.13/site-packages/mooncake/libasio.so",
    ):
        try:
            ctypes.CDLL(_dep, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

"""


def main():
    src = open(P).read()

    # 1. preload before the mooncake import (module-level, first occurrence)
    anchor_imp = "    from mooncake.engine import TransferEngine"
    if "PD fix: preload mooncake" not in src:
        assert anchor_imp in src, "import anchor missing"
        src = src.replace(anchor_imp, PRELOAD + anchor_imp, 1)
        print("1. preload applied")

    # 2+3. rewrite the WORKER-side register_kv_caches (LAST occurrence!)
    start = src.rindex("    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):")
    end = src.index("        self.device_kv_caches = kv_caches", start)
    new_fn = """    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        # Register the KV Cache data in mooncake (global dedup).

        logger.info("Registering KV_Caches. use_mla: %s", self.use_mla)

        kv_data_ptrs = []
        kv_data_lens = []
        self.block_len_per_layer = []

        split_k_and_v = self.transfer_topo.split_k_and_v
        tensor_size_bytes = None
        candidates = []
        for layer_name, cache_or_caches in kv_caches.items():
            if isinstance(cache_or_caches, (list, tuple)):
                cache_list = cache_or_caches  # hybrid KV: one tensor per group
            elif split_k_and_v:
                cache_list = cache_or_caches
            else:
                cache_list = [cache_or_caches]
            logger.debug(
                "registering layer %s with %d cache tensor(s)",
                layer_name,
                len(cache_list),
            )

            for cache in cache_list:
                self._log_debug_cache_registration(layer_name, cache)
                base_addr = cache.data_ptr()
                if tensor_size_bytes is None:
                    tensor_size_bytes = cache.nbytes
                    self.num_blocks = cache.shape[0]
                assert cache.shape[0] == self.num_blocks, (
                    "All kv cache tensors must have the same number of blocks"
                )
                block_len = cache.stride(0) * cache.element_size()
                candidates.append(
                    (base_addr, base_addr + self.num_blocks * block_len, block_len)
                )

        # The engine accepts only non-overlapping regions and checks only
        # consecutive pairs after sorting by address. Mirror that rule
        # GLOBALLY: hybrid KV views alias shared allocations across groups
        # and across the k/v cache dict keys.
        candidates.sort()
        last_end = None
        for addr, region_end, block_len in candidates:
            if last_end is not None and addr < last_end:
                continue
            last_end = region_end
            self.block_len_per_layer.append(block_len)
            kv_data_ptrs.append(addr)
            kv_data_lens.append(region_end - addr)

        self.kv_caches_base_addr = kv_data_ptrs
        self.seen_base_addresses = kv_data_ptrs

        ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
        if ret_value != 0:
            raise RuntimeError("Mooncake batch memory registration failed.")

        assert tensor_size_bytes is not None
        assert self.num_blocks != 0

"""
    src = src[:start] + new_fn + src[end:]

    # 4. PDX probes (remove once transfer works)
    probes = [
        ("    async def _start_load_kv(\n        self, reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]]\n    ):",
         "    async def _start_load_kv(\n        self, reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]]\n    ):\n        logger.info(\"PDX1 _start_load_kv reqs=%d\", len(reqs_to_recv))"),
        ("    def receive_kv(\n        self,\n        remote_engine_id: EngineId,\n        pull_metas: dict[ReqId, PullReqMeta],\n    ):",
         "    def receive_kv(\n        self,\n        remote_engine_id: EngineId,\n        pull_metas: dict[ReqId, PullReqMeta],\n    ):\n        logger.info(\"PDX2 receive_kv engine=%s metas=%d\", remote_engine_id, len(pull_metas))"),
        ("    async def receive_kv_from_single_worker(\n        self,\n        worker_addr: str,\n        pull_metas: dict[ReqId, PullReqMeta],\n    ):",
         "    async def receive_kv_from_single_worker(\n        self,\n        worker_addr: str,\n        pull_metas: dict[ReqId, PullReqMeta],\n    ):\n        logger.info(\"PDX3 recv_from_worker addr=%s metas=%d\", worker_addr, len(pull_metas))"),
    ]
    for old, new in probes:
        if "PDX1 _start_load_kv" in src and old.startswith("    async def _start_load_kv"):
            continue
        if "PDX2 receive_kv" in src and old.startswith("    def receive_kv"):
            continue
        if "PDX3 recv_from_worker" in src and old.startswith("    async def receive_kv_from_single_worker"):
            continue
        assert old in src, f"probe anchor missing: {old[:60]}"
        src = src.replace(old, new, 1)
    if "PDX4 send_kv_to_decode called" not in src:
        old4 = "    async def send_kv_to_decode(\n        self, identity: bytes, sock: zmq.asyncio.Socket, meta: MooncakeXferMetadata\n    ):\n        pending_reqs: dict[ReqId, SendBlockMeta] = {}"
        new4 = old4.replace("        pending_reqs:", "        logger.info(\"PDX4 send_kv_to_decode called\")\n        pending_reqs:")
        assert old4 in src, "PDX4 anchor missing"
        src = src.replace(old4, new4, 1)

    open(P, "w").write(src)
    import py_compile
    py_compile.compile(P, doraise=True)
    print("patches applied + syntax OK")


if __name__ == "__main__":
    sys.exit(main())

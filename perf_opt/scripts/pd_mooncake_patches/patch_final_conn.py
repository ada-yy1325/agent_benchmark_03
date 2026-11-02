import shutil
# 1. install the rebuilt engine lib
shutil.copy2(
    "/inspire/sj-ssd3/project/project-public/s26068/mooncake-src/build/mooncake-transfer-engine/src/libtransfer_engine.so",
    "/opt/mamba/lib/python3.13/site-packages/mooncake/libtransfer_engine.so",
)
print("ENGINE LIB INSTALLED")

# 2. restore the connector to pristine + preload + flatten only (no dedup)
P = "/opt/mamba/lib/python3.13/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"
shutil.copy2("/tmp/mc_0230.py", P)
src = open(P).read()

anchor_imp = "    from mooncake.engine import TransferEngine"
preload = """    # PD fix: preload mooncake Ascend build deps by absolute path (no LD_LIBRARY_PATH pollution)
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
assert anchor_imp in src
src = src.replace(anchor_imp, preload + anchor_imp, 1)

start = src.rindex("    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):")
end = src.index("        self.device_kv_caches = kv_caches", start)
new_fn = """    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        # Register the KV Cache data in mooncake (hybrid flatten only;
        # the engine accepts overlapped regions after the 2026-11-02 rebuild).

        logger.info("Registering KV_Caches. use_mla: %s", self.use_mla)

        kv_data_ptrs = []
        kv_data_lens = []
        self.block_len_per_layer = []

        split_k_and_v = self.transfer_topo.split_k_and_v
        tensor_size_bytes = None
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
                if base_addr in kv_data_ptrs:
                    continue
                if tensor_size_bytes is None:
                    tensor_size_bytes = cache.nbytes
                    self.num_blocks = cache.shape[0]
                assert cache.shape[0] == self.num_blocks, (
                    "All kv cache tensors must have the same number of blocks"
                )
                # Use stride-based block length so RDMA reaches the last
                # block's padding (e.g. DeepseekV4 MLA alignment). stride(0)
                # reflects the actual byte distance between consecutive
                # blocks in GPU memory, which matches or exceeds the
                # shape-based size.
                block_len = cache.stride(0) * cache.element_size()

                self.block_len_per_layer.append(block_len)
                kv_data_ptrs.append(base_addr)
                kv_data_lens.append(self.num_blocks * block_len)

        self.kv_caches_base_addr = kv_data_ptrs
        self.seen_base_addresses = kv_data_ptrs

        ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
        if ret_value != 0:
            raise RuntimeError("Mooncake batch memory registration failed.")

        assert tensor_size_bytes is not None
        assert self.num_blocks != 0

"""
src = src[:start] + new_fn + src[end:]
open(P, "w").write(src)
print("CONNECTOR REPATCHED")

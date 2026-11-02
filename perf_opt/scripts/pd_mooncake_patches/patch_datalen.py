p = "/opt/mamba/lib/python3.13/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"
s = open(p).read()

# 1. metadata field
old = "    kv_caches_base_addr: list[int]" + chr(10) + "    block_lens: list[int]" + chr(10)
new = old + "    kv_block_lens: list[int] = []" + chr(10)
assert old in s, "m1 missing"
s = s.replace(old, new, 1)

# 2. record per-group data lengths
old = "        self.block_len_per_layer = []" + chr(10) + "        self.kv_block_data_len_per_layer = []"
if old not in s:
    old2 = "        self.block_len_per_layer = []"
    assert old2 in s, "r1 missing"
    s = s.replace(old2, old, 1)
old = "                self.block_len_per_layer.append(block_len)" + chr(10) + "                kv_data_ptrs.append(base_addr)"
new = "                self.block_len_per_layer.append(block_len)" + chr(10) + "                self.kv_block_data_len_per_layer.append(cache.nbytes // cache.shape[0])" + chr(10) + "                kv_data_ptrs.append(base_addr)"
assert old in s, "r2 missing"
s = s.replace(old, new, 1)

# 3. _expand_transfer_regions takes data lengths
old = "        kv_block_len = block_len // 2 if is_kv_layout_blocks_first else block_len"
new = "        data_len = kv_block_lens[idx] if idx < len(kv_block_lens) else block_len" + chr(10) + "        kv_block_len = data_len // 2 if is_kv_layout_blocks_first else data_len"
assert old in s, "e1 missing"
s = s.replace(old, new, 1)
old = "    for base_addr, block_len in zip(base_addrs, block_lens):"
new = "    for idx, (base_addr, block_len) in enumerate(zip(base_addrs, block_lens)):"
assert old in s, "e2 missing"
s = s.replace(old, new, 1)
old = "    is_kv_layout_blocks_first: bool," + chr(10) + ") -> list[TransferRegion]:"
new = "    is_kv_layout_blocks_first: bool," + chr(10) + "    kv_block_lens: list[int] | None = None," + chr(10) + ") -> list[TransferRegion]:"
assert old in s, "e3 missing"
s = s.replace(old, new, 1)
old = "    regions: list[TransferRegion] = []"
new = "    if kv_block_lens is None:" + chr(10) + "        kv_block_lens = list(block_lens)" + chr(10) + "    regions: list[TransferRegion] = []"
assert old in s, "e4 missing"
s = s.replace(old, new, 1)

# 4. _get_transfer_regions passes data lengths
old = "        self, base_addrs: list[int], block_lens: list[int]" + chr(10) + "    ) -> list[TransferRegion]:"
new = "        self," + chr(10) + "        base_addrs: list[int]," + chr(10) + "        block_lens: list[int]," + chr(10) + "        kv_block_lens: list[int] | None = None," + chr(10) + "    ) -> list[TransferRegion]:"
assert old in s, "g1 missing"
s = s.replace(old, new, 1)
old = "            is_kv_layout_blocks_first=self.transfer_topo.virtually_split_kv_in_blocks," + chr(10) + "        )"
new = "            is_kv_layout_blocks_first=self.transfer_topo.virtually_split_kv_in_blocks," + chr(10) + "            kv_block_lens=kv_block_lens," + chr(10) + "        )"
assert old in s, "g2 missing"
s = s.replace(old, new, 1)

# 5. validation call sites
old = "        local_regions = self._get_transfer_regions(" + chr(10) + "            self.kv_caches_base_addr, self.block_len_per_layer" + chr(10) + "        )"
new = "        local_regions = self._get_transfer_regions(" + chr(10) + "            self.kv_caches_base_addr," + chr(10) + "            self.block_len_per_layer," + chr(10) + "            self.kv_block_data_len_per_layer," + chr(10) + "        )"
assert old in s, "v1 missing"
s = s.replace(old, new, 1)
old = "        remote_regions = self._get_transfer_regions(" + chr(10) + "            meta.kv_caches_base_addr, meta.block_lens" + chr(10) + "        )"
new = "        remote_regions = self._get_transfer_regions(" + chr(10) + "            meta.kv_caches_base_addr, meta.block_lens, meta.kv_block_lens" + chr(10) + "        )"
assert old in s, "v2 missing"
s = s.replace(old, new, 1)

# 6. metadata construction carries the data lengths
old = "            kv_caches_base_addr=self.kv_caches_base_addr," + chr(10) + "            block_lens=self.block_len_per_layer," + chr(10) + "        )"
new = "            kv_caches_base_addr=self.kv_caches_base_addr," + chr(10) + "            block_lens=self.block_len_per_layer," + chr(10) + "            kv_block_lens=self.kv_block_data_len_per_layer," + chr(10) + "        )"
assert old in s, "c1 missing"
s = s.replace(old, new, 1)

open(p, "w").write(s)
print("DATA-LENGTH SPLIT APPLIED")

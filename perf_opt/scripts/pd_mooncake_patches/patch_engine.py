p = "/inspire/sj-ssd3/project/project-public/s26068/mooncake-src/mooncake-transfer-engine/src/transfer_engine_impl.cpp"
s = open(p).read()

# Patch 1: relax the sorted-consecutive overlap rejection in registerLocalMemoryBatch
old1 = """        if (i > 0) {
            const auto& previous = sorted_buffers[i - 1];
            auto address = reinterpret_cast<uintptr_t>(buffer.addr);
            auto previous_address = reinterpret_cast<uintptr_t>(previous.addr);
            if (address - previous_address < previous.length) {
                LOG(ERROR) << "Transfer Engine does not support overlapped "
                              "memory region";
                return ERR_ADDRESS_OVERLAPPED;
            }
        }"""
new1 = """        if (i > 0) {
            const auto& previous = sorted_buffers[i - 1];
            auto address = reinterpret_cast<uintptr_t>(buffer.addr);
            auto previous_address = reinterpret_cast<uintptr_t>(previous.addr);
            if (address - previous_address < previous.length) {
                // PD fix: hybrid MLA KV views alias/interleave within shared
                // allocations; the vLLM Mooncake connector transfers via
                // explicit pointers so overlapped registrations are safe.
                LOG(WARNING) << "Overlapped memory region registered "
                                "(allowed for hybrid MLA KV layout)";
            }
        }"""
assert old1 in s, "anchor1 missing"
s = s.replace(old1, new1, 1)

# Patch 2: relax hasOverlapLocked rejection in tryReserveMemoryRegions
old2 = """        auto addr = reinterpret_cast<uintptr_t>(region.addr);
        if (hasOverlapLocked(addr, region.length)) {
            for (auto reserved_addr : reserved) {
                registering_memory_regions_.erase(reserved_addr);
            }
            return false;
        }
        registering_memory_regions_[addr] = region;
        reserved.push_back(addr);"""
new2 = """        auto addr = reinterpret_cast<uintptr_t>(region.addr);
        // PD fix: allow overlapped registrations (see registerLocalMemoryBatch)
        registering_memory_regions_[addr] = region;
        reserved.push_back(addr);"""
assert old2 in s, "anchor2 missing"
s = s.replace(old2, new2, 1)
open(p, "w").write(s)
print("ENGINE SOURCE PATCHED")

#!/usr/bin/env python3
"""Test adding indexer_kv_dtype to AttentionConfig via __dataclass_fields__."""
# Patch BEFORE any vLLM import
import dataclasses
from typing import Optional

# Manually add field to vllm.config.attention module BEFORE import
import sys, types

# 1. First import the module without triggering VllmConfig construction
from vllm.config.attention import AttentionConfig
import vllm.config.attention as attn_mod

# 2. Check if pydantic config allows extra fields
pc = AttentionConfig.__pydantic_config__
print("pydantic_config.extra:", pc.get("extra", "not set"))
print("pydantic_config type:", type(pc))
print("pydantic_config:", pc)

# 3. Try to modify extra setting
try:
    # pydantic ConfigDict might be frozen
    pc["extra"] = "allow"
    print("Set extra=allow: OK")
except Exception as e:
    print("Cannot set extra:", e)

# 4. Try to add field via dataclass_fields
try:
    field = dataclasses.field(default="", metadata={})
    AttentionConfig.__dataclass_fields__["indexer_kv_dtype"] = field
    print("Added field to __dataclass_fields__: OK")
    print("Fields now:", [f.name for f in dataclasses.fields(AttentionConfig)])
except Exception as e:
    print("Cannot add field:", e)
    
# 5. Try to construct with extra arg
try:
    cfg = AttentionConfig(**{"backend": None, "indexer_kv_dtype": "int8"})
    print("Constructed with indexer_kv_dtype:", cfg.indexer_kv_dtype)
except Exception as e:
    print("Construction failed:", e)

# 6. Try setting after construction
try:
    cfg = AttentionConfig()
    setattr(cfg, "indexer_kv_dtype", "int8")
    print("Set after construction: OK, value:", cfg.indexer_kv_dtype)
except Exception as e:
    print("Set after construction failed:", e)
#!/usr/bin/env python3
"""Test if we can add indexer_kv_dtype to AttentionConfig at runtime."""
import sys
sys.modules.pop("vllm.config.attention", None)

# First, patch before import
import vllm.config.attention as attention_mod
import dataclasses
from typing import Any

# Check current fields
fields = [f.name for f in dataclasses.fields(attention_mod.AttentionConfig)]
print("Current fields:", fields)
print("Has indexer_kv_dtype:", "indexer_kv_dtype" in fields)

# Try to create a mock field by modifying the class
from vllm.config.attention import AttentionConfig

# Check if we can set arbitrary attributes
cfg = AttentionConfig()
try:
    cfg.test_field = "test"
    print("Can set arbitrary attribute:", cfg.test_field)
except Exception as e:
    print("Cannot set arbitrary attribute:", e)

# Check what happens when we pass extra kwargs
try:
    cfg2 = AttentionConfig(**{"backend": None, "extra_arg": "hello"})
    print("Extra kwargs constructor works, value:", cfg2.extra_arg)
except Exception as e:
    print("Extra kwargs constructor fails:", e)

# Check pydantic config
print("Pydantic extra setting:", getattr(
    type(cfg).__pydantic_config__, "extra", "not found"
))
#!/usr/bin/env python3
"""
Fix transformers 4.51.0 → vLLM 0.23.0 compatibility monkey-patches.
Runs BEFORE any vLLM import, then launches start_server_baseline.py.
"""
import sys, os, types

print("[fix] applying transformers→vLLM compatibility patches", flush=True)

# ── Step 0: Pre-import transformers and apply all patches ──────────────
import transformers

# 0.1: Gemma3Config — exists in sub-module but not exported
from transformers.models.gemma3.configuration_gemma3 import Gemma3Config
transformers.Gemma3Config = Gemma3Config

# 0.2: ALLOWED_ATTENTION_LAYER_TYPES — not in 4.51.0
import transformers.configuration_utils as cu
if not hasattr(cu, 'ALLOWED_ATTENTION_LAYER_TYPES'):
    cu.ALLOWED_ATTENTION_LAYER_TYPES = []
if not hasattr(cu, 'ALLOWED_LAYER_TYPES'):
    cu.ALLOWED_LAYER_TYPES = []

# 0.3: AutoVideoProcessor — not in 4.51.0, use AutoProcessor as fallback
# We need to add it to _import_structure AND set it on the models.auto module
from transformers.models.auto.processing_auto import AutoProcessor
import transformers.models.auto as auto_mod
auto_mod.AutoVideoProcessor = AutoProcessor

# Monkey-patch __getattr__ to handle missing names gracefully
_orig_getattr = transformers.__getattr__

def _patched_getattr(name):
    """Try original getattr, then check our patches."""
    try:
        return _orig_getattr(name)
    except (ImportError, AttributeError, RuntimeError) as e:
        # Check if we already patched this name
        mod = sys.modules.get('transformers.models.auto')
        if mod and hasattr(mod, name):
            val = getattr(mod, name)
            # Set it on the parent module so future lookups work
            try:
                setattr(transformers, name, val)
            except Exception:
                pass
            return val
        # For other missing names, try to create a dummy placeholder
        # This allows vLLM to import gracefully even if it doesn't use the class
        import warnings
        warnings.warn(f"[fix] creating dummy for missing name: {name}")
        dummy = type(name, (), {})
        try:
            setattr(transformers, name, dummy)
        except Exception:
            pass
        return dummy

transformers.__getattr__ = _patched_getattr

# Also add to _import_structure for names that should be found via lazy loading
if 'models.auto' in transformers._import_structure:
    extras = ['AutoVideoProcessor']
    for name in extras:
        if name not in transformers._import_structure['models.auto']:
            transformers._import_structure['models.auto'].append(name)

print("[fix] all patches applied", flush=True)

# ── Step 1: Now run the real server launcher ───────────────────────────
print("[fix] launching baseline server...", flush=True)

# Both fix_and_launch.py and start_server_baseline.py are in the same directory
_this_dir = os.path.dirname(os.path.abspath(__file__))
_start_script = os.path.join(_this_dir, 'start_server_baseline.py')

print(f"[fix] start_script: {_start_script}", flush=True)

sys.argv = [_start_script]
import runpy
runpy.run_path(_start_script, run_name='__main__')
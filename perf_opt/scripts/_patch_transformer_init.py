#!/usr/bin/env python3
"""One-shot: patch transformers/__init__.py for vLLM 0.23.0 compat."""
import site, os

init_path = os.path.join(site.getsitepackages()[0], 'transformers', '__init__.py')
marker = '# === vLLM-0.23.0-compat-patch ==='

with open(init_path, 'r') as f:
    content = f.read()

if marker in content:
    print('Already patched')
    import sys; sys.exit(0)

# Shim to append to __init__.py
shim = marker + '''
import sys as _fsys

# Patch _import_structure for lazy loader
if '_import_structure' in dir():
    _istruct = {k: list(v) for k, v in _import_structure.items()}
    if 'models.auto' in _istruct:
        for _n in ['AutoVideoProcessor', 'Gemma3Config']:
            if _n not in _istruct['models.auto']:
                _istruct['models.auto'].append(_n)
    _import_structure = _istruct

try:
    import transformers.models.auto as _fauto
    if not hasattr(_fauto, 'AutoVideoProcessor'):
        from transformers.models.auto.processing_auto import AutoProcessor as _AVP
        _fauto.AutoVideoProcessor = _AVP
except Exception as e:
    print('[compat-shim] AutoVideoProcessor:', e)

try:
    import transformers as _ftrans
    if not hasattr(_ftrans, 'Gemma3Config'):
        from transformers.models.gemma3.configuration_gemma3 import Gemma3Config as _GC
        _ftrans.Gemma3Config = _GC
except Exception as e:
    print('[compat-shim] Gemma3Config:', e)

try:
    import transformers.configuration_utils as _fcu
    for _n in ['ALLOWED_ATTENTION_LAYER_TYPES', 'ALLOWED_LAYER_TYPES']:
        if not hasattr(_fcu, _n):
            setattr(_fcu, _n, [])
except Exception as e:
    print('[compat-shim] config_utils:', e)
'''
with open(init_path, 'a') as f:
    f.write(shim)
print('Patched:', init_path)
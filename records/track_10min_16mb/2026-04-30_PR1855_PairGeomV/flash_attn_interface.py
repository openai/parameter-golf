try:
    from hopper.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
except Exception:
    try:
        import importlib.util
        import os
        import sys

        _this = os.path.abspath(__file__)
        _loaded = None
        for _path in sys.path:
            _candidate = os.path.abspath(os.path.join(_path, "flash_attn_interface.py"))
            if _candidate == _this or not os.path.exists(_candidate):
                continue
            _spec = importlib.util.spec_from_file_location("_flash_attn_interface_external", _candidate)
            if _spec and _spec.loader:
                _loaded = importlib.util.module_from_spec(_spec)
                sys.modules[_spec.name] = _loaded
                _spec.loader.exec_module(_loaded)
                break
        if _loaded is None:
            raise ImportError("top-level flash_attn_interface not found")
        flash_attn_func = _loaded.flash_attn_func
        flash_attn_varlen_func = _loaded.flash_attn_varlen_func
    except Exception:
        from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func

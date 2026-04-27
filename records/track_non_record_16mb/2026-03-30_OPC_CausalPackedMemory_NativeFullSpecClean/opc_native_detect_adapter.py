from pathlib import Path
import sys

_VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"
if _VENDOR_ROOT.exists() and str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from opc_parameter_golf_submission.opc_native_detect_adapter import build_adapter


__all__ = ["build_adapter"]

import os
import sys
import tempfile
import types
import unittest
import urllib.error
from pathlib import Path
from unittest import mock


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

if "runpod_safe" not in sys.modules:
    sys.modules["runpod_safe"] = types.ModuleType("runpod_safe")

_rs = sys.modules["runpod_safe"]
_rs.UA = "test"
_rs.RUNTIME_WAIT_SECONDS = 1
_rs._make_ssl_ctx = lambda: None
_rs._ssh_upload = lambda *a, **kw: None
_rs.balance = lambda: (0.0, "USD")
_rs.create_pod = lambda **kw: {"id": "test"}
_rs.get_pods = lambda *a, **kw: []
_rs.terminate_and_wait = lambda *a, **kw: None
_rs.wait_runtime = lambda *a, **kw: {"uptimeInSeconds": 0}
_rs.GPU_SKU_TABLE = {}

sys.modules.pop("runpod_http_rehearsal", None)

import runpod_http_rehearsal as rhr


class _DummyResponse:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestDownloadFile(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    @mock.patch("runpod_http_rehearsal.time.sleep", return_value=None)
    def test_optional_transient_http_error_returns_none(self, _sleep):
        err = urllib.error.HTTPError(
            "https://example.invalid/foo.json", 502, "Bad Gateway", hdrs=None, fp=None
        )
        with mock.patch(
            "runpod_http_rehearsal.urllib.request.urlopen",
            side_effect=[err] * 6,
        ):
            result = rhr.download_file("pod", 30000, "foo.json", self.out_dir, optional=True)
        self.assertIsNone(result)
        self.assertFalse((self.out_dir / "foo.json").exists())
        self.assertEqual(_sleep.call_count, 5)

    @mock.patch("runpod_http_rehearsal.time.sleep", return_value=None)
    def test_optional_transient_http_error_can_recover(self, _sleep):
        err = urllib.error.HTTPError(
            "https://example.invalid/foo.json", 502, "Bad Gateway", hdrs=None, fp=None
        )
        with mock.patch(
            "runpod_http_rehearsal.urllib.request.urlopen",
            side_effect=[err, _DummyResponse(b"ok")],
        ):
            result = rhr.download_file("pod", 30000, "foo.json", self.out_dir, optional=True)
        self.assertEqual(result.read_bytes(), b"ok")
        self.assertEqual(_sleep.call_count, 1)

    @mock.patch("runpod_http_rehearsal.time.sleep", return_value=None)
    def test_required_transient_http_error_still_raises(self, _sleep):
        err = urllib.error.HTTPError(
            "https://example.invalid/foo.json", 502, "Bad Gateway", hdrs=None, fp=None
        )
        with mock.patch(
            "runpod_http_rehearsal.urllib.request.urlopen",
            side_effect=[err] * 6,
        ):
            with self.assertRaises(urllib.error.HTTPError):
                rhr.download_file("pod", 30000, "foo.json", self.out_dir, optional=False)
        self.assertEqual(_sleep.call_count, 5)

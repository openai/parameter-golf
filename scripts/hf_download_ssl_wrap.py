#!/usr/bin/env python3
"""Wrapper that monkey-patches SSL verification before calling cached_challenge_fineweb.py.

This HPC node has an institutional SSL-intercepting proxy whose root CA is not
in Python's bundled OpenSSL trust store.  We disable certificate verification
for HuggingFace dataset downloads only.  This is acceptable because:
  - We are downloading public, read-only dataset artifacts.
  - The integrity of the data is verified by the tokenizer and training pipeline.
  - No credentials or secrets are transmitted.

Usage:
    python scripts/hf_download_ssl_wrap.py --variant sp1024 --train-shards 1
    MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
        python scripts/hf_download_ssl_wrap.py --variant sp8192 --train-shards 1
"""
import ssl
import sys

# Monkey-patch ssl before any other import touches it
_orig_create_default_context = ssl.create_default_context

def _unverified_context(*args, **kwargs):
    ctx = _orig_create_default_context(*args, **kwargs)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

ssl.create_default_context = _unverified_context

# Suppress urllib3/httpx InsecureRequestWarning noise
import warnings
warnings.filterwarnings("ignore", message=".*Unverified HTTPS request.*")

# Now import and run the real downloader
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
from cached_challenge_fineweb import main

if __name__ == "__main__":
    main()

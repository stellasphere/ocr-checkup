from __future__ import annotations

import hashlib
import os
import time


def _new_id(prefix: str) -> str:
    now = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rand = os.urandom(8)
    digest = hashlib.sha256(rand).hexdigest()[:8]
    return f"{prefix}_{now}_{digest}"


def new_run_id() -> str:
    return _new_id("run")


def new_evaluation_id() -> str:
    return _new_id("eval")


__all__ = [
    "new_run_id",
    "new_evaluation_id",
]

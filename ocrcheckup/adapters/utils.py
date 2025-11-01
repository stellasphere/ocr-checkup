from __future__ import annotations

from threading import Lock
from typing import Dict, Tuple

from ocrcheckup.rate_limiter import RateLimiter


_limiters: Dict[Tuple[str, int], RateLimiter] = {}
_lock = Lock()


def rate_limit(adapter_id: str, rpm: int) -> None:
    if rpm <= 0:
        raise ValueError("rpm must be positive")
    key = (adapter_id, rpm)
    with _lock:
        limiter = _limiters.get(key)
        if limiter is None:
            limiter = RateLimiter(rpm)
            _limiters[key] = limiter
    limiter.wait_if_needed()


__all__ = ["rate_limit"]

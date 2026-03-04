from __future__ import annotations

import time
import threading
from collections import deque


class RateLimiter:
    """Simple sliding-window rate limiter. Thread-safe."""

    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self.max_requests = requests_per_minute
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def wait_if_needed(self) -> None:
        """Block until a request slot is available within the rate limit window."""
        with self._lock:
            now = time.monotonic()
            # Evict timestamps older than 60 seconds
            while self._timestamps and self._timestamps[0] <= now - 60.0:
                self._timestamps.popleft()

            if len(self._timestamps) >= self.max_requests:
                # Wait until the oldest timestamp exits the window
                sleep_until = self._timestamps[0] + 60.0
                wait = sleep_until - now
                if wait > 0:
                    self._lock.release()
                    time.sleep(wait)
                    self._lock.acquire()
                    # Re-evict after sleeping
                    now = time.monotonic()
                    while self._timestamps and self._timestamps[0] <= now - 60.0:
                        self._timestamps.popleft()

            self._timestamps.append(time.monotonic())

import time
from collections import deque
import threading

class RateLimiter:
    def __init__(self, requests_per_minute: int):
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self.max_requests = requests_per_minute
        # Calculate the minimum interval between requests in seconds
        self.min_interval = 60.0 / self.max_requests
        # Use a deque to store timestamps of recent requests
        self.request_timestamps = deque()
        # Lock for thread safety
        self._lock = threading.Lock()

    def wait_if_needed(self):
        with self._lock:
            while True:
                now = time.monotonic()
                while self.request_timestamps and self.request_timestamps[0] <= now - 60.0:
                    self.request_timestamps.popleft()

                if len(self.request_timestamps) < self.max_requests:
                    # Enough capacity, no wait needed, or first few requests
                    self.request_timestamps.append(now)
                    break
                else:
                    # We are at capacity. Calculate time until the oldest request
                    time_since_oldest_allowed = now - self.request_timestamps[0]
                    required_wait = self.min_interval - (now - self.request_timestamps[-self.max_requests]) # Time until the slot opens up

                    next_allowed_time = self.request_timestamps[-self.max_requests + 1] + self.min_interval
                    next_allowed_time_precise = self.request_timestamps[0] + 60.0 # Time when the oldest timestamp expires
                    wait_duration = max(0, self.min_interval - (now - self.request_timestamps[-1]) if self.request_timestamps else 0) # Wait based on last request
                    wait_duration_window = max(0, (self.request_timestamps[0] + 60.0) - now if len(self.request_timestamps) >= self.max_requests else 0) # Wait based on window

                    earliest_next_time = self.request_timestamps[-self.max_requests +1] + self.min_interval if len(self.request_timestamps) >= self.max_requests else now

                    # Check if the earliest time we can run is in the future
                    wait_needed = earliest_next_time - now
                    if wait_needed > 0:
                         self._lock.release()
                         time.sleep(wait_needed)
                         self._lock.acquire()
                         continue # Loop back to re-check conditions after waking up
                    else:
                         self.request_timestamps.append(now)
                         break # Break the inner while loop

            now = time.monotonic()
            while self.request_timestamps and self.request_timestamps[0] <= now - 60.0:
                self.request_timestamps.popleft() 
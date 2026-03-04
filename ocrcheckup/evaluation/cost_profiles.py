from __future__ import annotations

from typing import Dict


# Built-in compute cost profiles: name -> USD per hour
COST_PROFILES: Dict[str, float] = {
    "free": 0.0,
    "macos-m1": 0.0,
    "macos-m2": 0.0,
    "macos-m3": 0.0,
    "macos-m4": 0.0,
    "t4-gcp": 0.35,
    "l4-gcp": 0.81,
    "a100-40gb-gcp": 3.67,
    "a100-80gb-gcp": 5.07,
    "h100-gcp": 12.12,
}


def get_cost_profile(name: str) -> float:
    """Return the USD/hr rate for a named compute cost profile."""
    if name not in COST_PROFILES:
        raise KeyError(
            f"Unknown cost profile: {name!r}. "
            f"Available: {sorted(COST_PROFILES.keys())}"
        )
    return COST_PROFILES[name]


def compute_local_cost(elapsed_seconds: float, usd_per_hour: float) -> float:
    """Compute cost for local model inference based on wall-clock time."""
    return (elapsed_seconds / 3600.0) * usd_per_hour

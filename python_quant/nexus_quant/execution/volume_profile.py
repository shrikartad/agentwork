"""Empirical data-driven intraday volume profiling and forecasting.

Phase 4 / Priority 3 execution realism:
Replaces heuristic / analytical volume curves (e.g. cosine U-curves) with
calibrated empirical volume profiles derived from historical market data.

Properties guaranteed:
1. Leak-free walk-forward filtering: Day D forecast strictly uses dates < D.
2. Monotonicity: Cumulative fraction C(t) is continuous, monotonic non-decreasing,
   with C(0) = 0.0 and C(1) = 1.0.
3. Normalization: Discrete bucket weights sum to 1.0.
4. Robustness: Laplace smoothing / minimum floor prevents zero-volume division.
5. Symbol specificity: Profiles are keyed and tracked per symbol.
6. Deterministic JSON serialization for reproducibility.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class VolumeProfile:
    """Discrete intraday volume profile and cumulative schedule."""

    symbol: str
    bucket_weights: list[float]
    n_buckets: int = field(init=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.bucket_weights:
            raise ValueError("bucket_weights cannot be empty")
        weights = np.asarray(self.bucket_weights, dtype=np.float64)
        if np.any(weights < 0):
            raise ValueError("bucket_weights must be non-negative")
        total = float(np.sum(weights))
        if total <= 0:
            raise ValueError("sum of bucket_weights must be positive")
        # Normalize strictly to sum to 1.0
        normalized = (weights / total).tolist()
        self.bucket_weights = [float(w) for w in normalized]
        self.n_buckets = len(self.bucket_weights)

    def cumulative_fraction(self, t_frac: float) -> float:
        """Evaluate cumulative fraction C(t) in [0, 1] monotonically.

        Continuous piecewise linear interpolation through the bucket boundaries.
        C(0.0) = 0.0, C(1.0) = 1.0.
        """
        t = min(1.0, max(0.0, float(t_frac)))
        if t <= 0.0:
            return 0.0
        if t >= 1.0:
            return 1.0

        # Bucket index and position within the bucket
        pos = t * self.n_buckets
        idx = int(pos)
        if idx >= self.n_buckets:
            return 1.0

        cum_edges = np.zeros(self.n_buckets + 1, dtype=np.float64)
        cum_edges[1:] = np.cumsum(self.bucket_weights)
        cum_edges[-1] = 1.0

        frac_in_bucket = pos - idx
        c_val = cum_edges[idx] + frac_in_bucket * (cum_edges[idx + 1] - cum_edges[idx])
        return float(np.clip(c_val, 0.0, 1.0))

    def target_volume(self, t_frac: float, total_volume: float) -> float:
        """Target execution volume completed by t_frac."""
        return self.cumulative_fraction(t_frac) * max(0.0, float(total_volume))

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "bucket_weights": self.bucket_weights,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VolumeProfile:
        return cls(
            symbol=str(data["symbol"]),
            bucket_weights=[float(w) for w in data["bucket_weights"]],
            metadata=dict(data.get("metadata", {})),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> VolumeProfile:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)


class EmpiricalVolumeForecaster:
    """Historical volume curve aggregator and forecaster.

    Strictly maintains walk-forward temporal hygiene:
    For any target day D, only observations from dates strictly earlier than D
    (d < D) are included in the forecast.
    """

    def __init__(self, n_buckets: int = 26) -> None:
        """n_buckets: Number of intraday buckets (e.g. 26 15-minute buckets for 6.5h session)."""
        if n_buckets <= 0:
            raise ValueError("n_buckets must be positive")
        self.n_buckets = n_buckets
        # {symbol: {date_str_or_int: np.ndarray(n_buckets)}}
        self._history: dict[str, dict[Any, np.ndarray]] = {}

    def add_daily_volume(
        self,
        symbol: str,
        date: str | int,
        bucket_volumes: Sequence[float],
    ) -> None:
        """Record an observed full-day intraday volume vector."""
        sym = str(symbol).upper()
        vols = np.asarray(bucket_volumes, dtype=np.float64)
        if len(vols) != self.n_buckets:
            raise ValueError(f"Expected {self.n_buckets} buckets, got {len(vols)}")
        if np.any(vols < 0):
            raise ValueError("Bucket volumes cannot be negative")
        if sym not in self._history:
            self._history[sym] = {}
        self._history[sym][date] = vols.copy()

    def available_dates(self, symbol: str) -> list[Any]:
        sym = str(symbol).upper()
        if sym not in self._history:
            return []
        return sorted(self._history[sym].keys())

    def forecast(
        self,
        symbol: str,
        as_of_date: str | int,
        history_window: int = 5,
        smoothing_floor: float = 1e-4,
    ) -> VolumeProfile:
        """Compute an empirical volume profile for `as_of_date` using prior history.

        Args:
            symbol: Target ticker symbol.
            as_of_date: Target execution date. Only dates < as_of_date are used.
            history_window: Maximum number of recent prior sessions to average.
            smoothing_floor: Regularization epsilon added to each bucket to prevent
                zero-density bins or overfitting to quiet buckets.
        """
        sym = str(symbol).upper()
        if sym not in self._history:
            raise ValueError(f"No history recorded for symbol '{sym}'")

        # Strict walk-forward temporal filter: only dates strictly before as_of_date
        prior_dates = sorted([d for d in self._history[sym] if d < as_of_date])
        if not prior_dates:
            raise ValueError(
                f"No prior history available for '{sym}' strictly before '{as_of_date}' "
                f"(available: {sorted(self._history[sym].keys())})"
            )

        window_dates = prior_dates[-history_window:]
        daily_fractions = []
        for d in window_dates:
            raw = self._history[sym][d]
            tot = float(np.sum(raw))
            if tot > 0:
                daily_fractions.append(raw / tot)
            else:
                # Uniform fallback if day had zero recorded volume
                daily_fractions.append(np.full(self.n_buckets, 1.0 / self.n_buckets))

        # Average normalized intraday profiles
        avg_profile = np.mean(daily_fractions, axis=0)

        # Apply smoothing floor / regularization
        regularized = avg_profile + max(0.0, float(smoothing_floor))
        norm_weights = regularized / np.sum(regularized)

        metadata = {
            "symbol": sym,
            "as_of_date": as_of_date,
            "history_window": history_window,
            "dates_used": [str(d) for d in window_dates],
            "smoothing_floor": smoothing_floor,
            "n_history_days_used": len(window_dates),
        }

        return VolumeProfile(
            symbol=sym,
            bucket_weights=norm_weights.tolist(),
            metadata=metadata,
        )

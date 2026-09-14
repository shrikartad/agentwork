"""Bootstrap acceleration must preserve every sampled rank and CI."""

from __future__ import annotations

import numpy as np
import pytest
from nexus_quant.research.experiments import (
    _block_indices,
    _rank,
    _rank_ic_bootstrap,
    _resampled_ranks,
    rank_ic,
)


def _reference(y, pred, n_boot, seed, block, alpha):
    n = len(y)
    rng = np.random.default_rng(seed)
    length = block or max(1, int(np.sqrt(n)))
    count = (n + length - 1) // length
    samples = [rank_ic(y[idx], pred[idx]) for idx in (
        _block_indices(rng.integers(0, n, size=count), length, n) for _ in range(n_boot)
    )]
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return {"lo": float(lo), "hi": float(hi), "mean": float(np.mean(samples))}


def test_resampled_ranks_account_for_ties_and_missing_values():
    values = np.array([7.0, -4.0, 7.0, 99.0, 0.0])
    unique, codes = np.unique(values, return_inverse=True)
    idx = np.array([0, 0, 0, 2, 4, 4, 1])
    np.testing.assert_array_equal(_resampled_ranks(codes[idx], len(unique)), _rank(values[idx]))
    assert not np.array_equal(_rank(values)[idx], _rank(values[idx]))


@pytest.mark.parametrize("kind", ("continuous", "discrete", "constant", "nonfinite"))
@pytest.mark.parametrize("block", (0, 1, 7, 100))
@pytest.mark.parametrize("alpha", (0.05, 0.2))
def test_fast_bootstrap_exactly_matches_sort_each_draw(kind, block, alpha):
    rng = np.random.default_rng(319)
    y = rng.integers(-3, 4, size=61).astype(float)
    pred = rng.normal(size=y.size)
    if kind == "discrete":
        pred = pred.round()
    elif kind == "constant":
        pred[:] = 1.0
    elif kind == "nonfinite":
        pred[[2, 10, 13]] = [np.nan, np.nan, np.inf]
    kwargs = {"n_boot": 60, "seed": 321, "block": block, "alpha": alpha}
    expected = _reference(y, pred, **kwargs)
    actual = _rank_ic_bootstrap(y, pred, **kwargs)
    np.testing.assert_array_equal(list(actual.values()), list(expected.values()))

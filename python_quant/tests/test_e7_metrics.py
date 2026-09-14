"""Hand-constructed test cases for E7 execution metrics: max_drawdown and fill_rate."""

from __future__ import annotations

import numpy as np
import pytest
from nexus_quant.agents.evaluate import (
    _episode_rows,
    _family_bootstrap_statistics,
    ci_from_rows,
    evaluate_regime_ci,
    paired_difference_ci,
    run_regime_episodes,
)
from nexus_quant.envs.order_book_env import OrderBookEnv
from nexus_quant.execution.metrics import fill_rate, max_drawdown


# ---------------------------------------------------------------------------
# max_drawdown hand-constructed tests
# ---------------------------------------------------------------------------
def test_max_drawdown_strictly_increasing():
    # Never draws down; peak updates at every step
    path = [0.0, 5.0, 10.0, 15.0, 20.0]
    assert max_drawdown(path) == 0.0


def test_max_drawdown_strictly_decreasing():
    # Trough is at the very end
    path = [20.0, 15.0, 10.0, 5.0, 0.0]
    assert max_drawdown(path) == pytest.approx(20.0)


def test_max_drawdown_multi_peak_known():
    # Peak reaches 20, drops to 8 -> drawdown 12; later recovery to 15 does not exceed peak
    path = [0.0, 10.0, 5.0, 20.0, 8.0, 15.0]
    assert max_drawdown(path) == pytest.approx(12.0)

    # First peak 120 -> drop to 70 (dd 50). Later new peak 130 -> drop to 90 (dd 40).
    path2 = [100.0, 120.0, 80.0, 110.0, 70.0, 130.0, 90.0]
    assert max_drawdown(path2) == pytest.approx(50.0)


def test_max_drawdown_negative_paths():
    # Peak is -10, trough is -30 -> drawdown is 20
    path = [-10.0, -20.0, -15.0, -30.0]
    assert max_drawdown(path) == pytest.approx(20.0)


def test_max_drawdown_edge_cases():
    assert max_drawdown([]) == 0.0
    assert max_drawdown([42.0]) == 0.0
    assert max_drawdown([10.0, 10.0, 10.0]) == 0.0


# ---------------------------------------------------------------------------
# fill_rate hand-constructed tests
# ---------------------------------------------------------------------------
def test_fill_rate_boundaries_and_empty():
    assert fill_rate([], 100) == 0.0
    assert fill_rate([(100, 50)], 0) == 0.0
    assert fill_rate([(100, 50)], -10) == 0.0


def test_fill_rate_partial_and_multi_slice():
    # Single slice: 50 / 200 = 0.25
    assert fill_rate([(100, 50)], 200) == pytest.approx(0.25)

    # Multi-slice 3-tuple format (t, px, sz): 50 + 50 = 100 / 200 = 0.5
    assert fill_rate([(0, 100, 50), (1, 101, 50)], 200) == pytest.approx(0.5)


def test_fill_rate_complete_and_overfill_clamp():
    # Exact completion
    assert fill_rate([(100, 200)], 200) == pytest.approx(1.0)

    # Overfill safety clamp
    assert fill_rate([(100, 250)], 200) == pytest.approx(1.0)


def test_fill_rate_inventory_conservation():
    """Verify fill_rate vs completion mathematical equivalence when inventory is conserved."""
    env = OrderBookEnv(seed=123, inventory=50)
    _, _ = env.reset(seed=123)
    # Take an aggressive market sell action to force fills
    for _ in range(5):
        _, _, term, trunc, _ = env.step(-1.0)
        if term or trunc:
            break

    total_filled = sum(sz for _, _, sz in env.fills)
    assert env.inventory0 - env.inventory == total_filled

    fr = fill_rate(env.fills, env.inventory0)
    completion = 1.0 - env.inventory / env.inventory0
    assert fr == pytest.approx(completion)


def test_episode_rows_includes_e7_metrics():
    """Verify _episode_rows exposes fill_rate and mdd_ticks."""
    env = OrderBookEnv(seed=7)
    rows = _episode_rows(env, lambda _env, _ob: -0.5, seeds=[7, 8])
    assert len(rows) == 2
    for r in rows:
        assert "fill_rate" in r
        assert "mdd_ticks" in r
        assert 0.0 <= r["fill_rate"] <= 1.0
        assert r["mdd_ticks"] >= 0.0


def test_episode_drawdown_includes_first_step_loss():
    env = OrderBookEnv(inventory=20, horizon=1, seed=7)
    row = _episode_rows(env, lambda _env, _ob: -1.0, seeds=[7])[0]
    expected = -env.mark_to_market()
    assert expected > 0.0
    assert row["mdd_ticks"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# E7 seed-family confidence intervals
# ---------------------------------------------------------------------------
class _ConstPolicy:
    def __init__(self, action: float) -> None:
        self.action = action

    def act(self, _obs, *, deterministic: bool = True) -> float:
        assert deterministic
        return self.action


def _paired_rows(metric: str, agent_value: float, baseline_value: float) -> dict:
    """Three equal seed families with two paired episodes each."""
    agent, baseline = [], []
    for family in range(3):
        for episode in range(2):
            seed = 10_000 * family + episode
            agent.append({"seed_family": family, "seed": seed, metric: agent_value})
            baseline.append({"seed_family": family, "seed": seed, metric: baseline_value})
    return {"synthetic": {"agent": agent, "baseline": baseline}}


@pytest.mark.parametrize("metric", ("fill_rate", "mdd_ticks"))
def test_e7_regime_cis_cover_both_metrics_across_seed_families_and_regimes(metric):
    """The public E7 APIs stay finite over three seed families and two regimes."""
    regimes = {
        "calm": lambda: OrderBookEnv(inventory=50, horizon=6, seed=7),
        "eventful": lambda: OrderBookEnv(
            inventory=50,
            horizon=6,
            seed=7,
            regime_prob=0.8,
            gap_prob=0.2,
            gap_min=10,
            gap_max=20,
            vol_events_min=1,
            vol_events_max=2,
            vol_take_min=4,
            vol_take_max=12,
        ),
    }
    policy = _ConstPolicy(-0.4)
    summaries = evaluate_regime_ci(
        policy,
        regimes,
        seeds=3,
        episodes_per_seed=2,
        baselines=("twap",),
        agent_name="agent",
        metric=metric,
        n_boot=80,
    )
    episodes = run_regime_episodes(
        policy,
        regimes,
        seeds=3,
        episodes_per_seed=2,
        baselines=("twap",),
        agent_name="agent",
    )
    paired = paired_difference_ci(
        policy,
        "twap",
        regimes,
        metric=metric,
        n_boot=80,
        episodes=episodes,
        agent_name="agent",
    )

    assert set(summaries) == set(regimes) == set(paired)
    for regime in regimes:
        for strategy in ("agent", "twap"):
            ci = summaries[regime][strategy]
            assert ci.metric == metric
            assert (ci.n_seeds, ci.n_episodes) == (3, 6)
            assert np.isfinite((ci.mean, *ci.ci95)).all()
            assert ci.ci95[0] <= ci.mean <= ci.ci95[1]
        result = paired[regime]
        assert result["n"] == 6.0
        assert np.isfinite([result["mean"], result["lo"], result["hi"], result["frac_agent_better"]]).all()
        assert result["lo"] <= result["mean"] <= result["hi"]
        assert result["pct_vs_baseline"] is None or np.isfinite(result["pct_vs_baseline"])


def test_seed_family_bootstrap_is_deterministic_and_resamples_whole_families():
    """A draw is the mean of full family copies, not an episode-level block."""
    families = [
        np.array([0.0, 2.0]),
        np.array([20.0, 26.0]),
        np.array([200.0, 210.0]),
    ]
    observed = _family_bootstrap_statistics(families, n_boot=25, seed=17)
    selected = np.random.default_rng(17).integers(0, 3, size=(25, 3))
    expected = np.array([
        np.concatenate([families[index] for index in draw]).mean() for draw in selected
    ])
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(
        observed,
        _family_bootstrap_statistics(families, n_boot=25, seed=17),
    )

    rows = [
        {"seed_family": family, "x": value}
        for family, values in ((2, families[2]), (0, families[0]), (1, families[1]))
        for value in values
    ]
    ci = ci_from_rows(rows, metric="x", name="s", regime="r", n_boot=80)
    assert ci.per_seed_mean == [1.0, 23.0, 205.0]
    assert ci.ci95[0] <= ci.mean <= ci.ci95[1]


def test_seed_family_bootstrap_rejects_unequal_family_sizes():
    rows = [
        {"seed_family": 0, "fill_rate": 0.1},
        {"seed_family": 0, "fill_rate": 0.2},
        {"seed_family": 1, "fill_rate": 0.3},
    ]
    with pytest.raises(ValueError, match="equal episode counts"):
        ci_from_rows(rows, metric="fill_rate", name="agent", regime="r", n_boot=20)


def test_seed_family_ci_rejects_one_family_instead_of_false_precision():
    rows = [{"seed_family": 0, "fill_rate": value} for value in (0.1, 0.9)]
    with pytest.raises(ValueError, match="at least 2 independent families"):
        ci_from_rows(rows, metric="fill_rate", name="agent", regime="r", n_boot=20)


def test_paired_ci_rejects_one_family_instead_of_false_precision():
    with pytest.raises(ValueError, match="at least 2 independent families"):
        paired_difference_ci(
            _ConstPolicy(-1.0), "twap", {"calm": OrderBookEnv},
            seeds=1, episodes_per_seed=2, metric="fill_rate", n_boot=20,
        )


def test_seed_family_bootstrap_rejects_nonintegral_family_identifiers():
    rows = [
        {"seed_family": 0, "fill_rate": 0.1},
        {"seed_family": 0.5, "fill_rate": 0.2},
    ]
    with pytest.raises(ValueError, match="seed_family must be an integer"):
        ci_from_rows(rows, metric="fill_rate", name="agent", regime="r", n_boot=20)


@pytest.mark.parametrize(
    ("metric", "agent_value", "baseline_value", "expected_delta"),
    (
        ("fill_rate", 0.75, 0.50, 0.25),  # higher fill fraction is better
        ("mdd_ticks", 2.0, 10.0, 8.0),    # lower drawdown is better
    ),
)
def test_paired_e7_metric_direction_and_seed_alignment(
    metric, agent_value, baseline_value, expected_delta,
):
    episodes = _paired_rows(metric, agent_value, baseline_value)
    episodes["synthetic"]["baseline"].reverse()  # pairing is by family + seed, not list position
    result = paired_difference_ci(
        _ConstPolicy(0.0),
        "baseline",
        {"synthetic": lambda: OrderBookEnv()},
        metric=metric,
        n_boot=80,
        episodes=episodes,
        agent_name="agent",
    )["synthetic"]
    assert result["mean"] == pytest.approx(expected_delta)
    assert result["frac_agent_better"] == 1.0
    assert result["pct_vs_baseline"] == pytest.approx(expected_delta / baseline_value * 100.0)
    assert result["lo"] <= result["mean"] <= result["hi"]

    bad = _paired_rows(metric, agent_value, baseline_value)
    bad["synthetic"]["baseline"][0]["seed_family"] = 99
    with pytest.raises(ValueError, match=r"identical \(seed_family, seed\) rows"):
        paired_difference_ci(
            _ConstPolicy(0.0),
            "baseline",
            {"synthetic": lambda: OrderBookEnv()},
            metric=metric,
            episodes=bad,
            agent_name="agent",
        )

    duplicate = _paired_rows(metric, agent_value, baseline_value)
    duplicate["synthetic"]["baseline"][1]["seed"] = duplicate["synthetic"]["baseline"][0]["seed"]
    with pytest.raises(ValueError, match=r"unique \(seed_family, seed\) rows"):
        paired_difference_ci(
            _ConstPolicy(0.0),
            "baseline",
            {"synthetic": lambda: OrderBookEnv()},
            metric=metric,
            episodes=duplicate,
            agent_name="agent",
        )

    nonintegral = _paired_rows(metric, agent_value, baseline_value)
    nonintegral["synthetic"]["baseline"][0]["seed"] = 0.5
    with pytest.raises(ValueError, match="seed must be an integer"):
        paired_difference_ci(
            _ConstPolicy(0.0),
            "baseline",
            {"synthetic": lambda: OrderBookEnv()},
            metric=metric,
            episodes=nonintegral,
            agent_name="agent",
        )

    uneven = _paired_rows(metric, agent_value, baseline_value)
    for strategy in ("agent", "baseline"):
        uneven["synthetic"][strategy].pop()
    with pytest.raises(ValueError, match="equal episode counts"):
        paired_difference_ci(
            _ConstPolicy(0.0),
            "baseline",
            {"synthetic": lambda: OrderBookEnv()},
            metric=metric,
            episodes=uneven,
            agent_name="agent",
        )


@pytest.mark.parametrize(
    ("metric", "agent_value", "baseline_value", "constant"),
    (
        ("fill_rate", 0.5, 0.0, 0.0),
        ("mdd_ticks", 0.0, 0.0, 0.0),
    ),
)
def test_e7_constant_and_zero_baseline_outputs_are_finite_without_percent_claim(
    metric, agent_value, baseline_value, constant,
):
    rows = [
        {"seed_family": family, metric: constant}
        for family in range(3)
        for _ in range(2)
    ]
    ci = ci_from_rows(rows, metric=metric, name="baseline", regime="r", n_boot=40)
    assert ci.mean == ci.ci95[0] == ci.ci95[1] == constant

    result = paired_difference_ci(
        _ConstPolicy(0.0),
        "baseline",
        {"synthetic": lambda: OrderBookEnv()},
        metric=metric,
        n_boot=40,
        episodes=_paired_rows(metric, agent_value, baseline_value),
        agent_name="agent",
    )["synthetic"]
    assert np.isfinite([result["mean"], result["lo"], result["hi"]]).all()
    assert result["lo"] <= result["mean"] <= result["hi"]
    assert result["pct_vs_baseline"] is None


@pytest.mark.parametrize("metric", ("fill_rate", "mdd_ticks"))
def test_e7_cis_reject_nonfinite_episode_metrics(metric):
    rows = [
        {"seed_family": family, metric: value}
        for family, value in ((0, 0.0), (1, float("nan")))
    ]
    with pytest.raises(ValueError, match="must be finite"):
        ci_from_rows(rows, metric=metric, name="agent", regime="r", n_boot=20)


@pytest.mark.parametrize("percentages", [[None], [None, 10.0], [float("nan")]])
def test_fairness_report_handles_undefined_percentages(percentages):
    from scripts.rl_fairness_study import render_markdown

    estimate = {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    summary = {"ppo_pooled_mean": 0.0, "ppo_seed_std": 0.0, "baselines": {"twap": estimate}}
    metrics = ("shortfall_bps", "completion", "fill_rate", "mdd_ticks", "vwap_slip_bps")
    result = {
        "config": {"train_regime": "calm", "holdout_regimes": [], "train_seeds": 2,
                   "eval_seeds": 2, "episodes_per_seed": 2, "iters": 0, "episodes": 2},
        "modes": {"novol": {
            "per_metric": {metric: {"calm": summary} for metric in metrics},
            "paired": {"calm": {
                "vs_best": [estimate],
                "vs_vwap": [{**estimate, "pct_vs_baseline": value} for value in percentages],
                "best_baseline": "twap", "best_baseline_mean": 0.0,
            }},
        }},
    }
    report = render_markdown(result)
    assert "bps (n/a)" in report
    assert "nan" not in report.lower()

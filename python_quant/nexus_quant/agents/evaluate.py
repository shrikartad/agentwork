"""Evaluation harness: trained PPO agent vs. the execution baselines.

The headline resume metric is **lower implementation-shortfall slippage than
VWAP**. ``shortfall_bps`` (from ``OrderBookEnv`` info) is ``(arrival_mid −
vwap) / arrival_mid × 1e4`` — how much the child execution conceded relative
to the arrival mid, in basis points. Lower is better.

Every strategy runs the same episode seeds and exogenous arrival descriptors.
Realized prices and fills remain endogenous to the strategy's book impact;
paired seeds do not imply identical realized trade tapes. Price and PnL metrics
are gross, while the environment's reward separately applies execution charges.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from ..envs.order_book_env import OrderBookEnv

BaselineId = Literal[
    "twap", "vwap", "pov", "passive",
    "schedule_twap", "adaptive_pov", "is_aware",
]

MetricDirection = Literal["lower", "higher"]

# Positive paired deltas mean improvement, including higher fill fractions.
_METRIC_DIRECTIONS: dict[str, MetricDirection] = {
    "shortfall_bps": "lower",
    "is_bps": "lower",
    "vwap_slip_bps": "lower",
    "leftover": "lower",
    "mdd_ticks": "lower",
    "reward": "higher",
    "completion": "higher",
    "fill_rate": "higher",
}


class Policy(Protocol):
    """Anything with ``act(obs, deterministic=True) -> float`` (PPOPolicy)."""

    def act(self, obs: np.ndarray, *, deterministic: bool = True) -> float: ...


@dataclass
class EvalSummary:
    name: str
    reward_mean: float
    shortfall_bps_mean: float
    shortfall_bps_std: float
    leftover_mean: float
    n: int


def _episode_shortfall(
    env: OrderBookEnv, act_fn: Callable[[np.ndarray], float], seed: int
) -> tuple[float, float, int]:
    """One full episode: reset(seed), then follow ``act_fn`` to the end."""
    obs, _ = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float64)
    total = 0.0
    while True:
        a = act_fn(obs)
        obs, r, term, trunc, info = env.step(a)
        obs = np.asarray(obs, dtype=np.float64)
        total += float(r)
        if term or trunc:
            return total, float(info["shortfall_bps"]), int(env.inventory)


def evaluate_policy(
    policy: Policy,
    *,
    n_episodes: int = 50,
    seed: int = 0,
    env_factory: Callable[[], OrderBookEnv] = OrderBookEnv,
    deterministic: bool = True,
) -> tuple[list[dict], EvalSummary]:
    """Run ``policy`` over ``n_episodes`` seeded episodes; return rows + summary.

    Rows are dicts for easy tabulation; the summary carries the mean
    shortfall (the slippage metric) and its scatter.
    """
    env = env_factory()
    rows: list[dict] = []
    rewards: list[float] = []
    sfs: list[float] = []
    leftovers: list[int] = []
    for i in range(n_episodes):
        total, sf, leftover = _episode_shortfall(
            env, lambda ob: policy.act(ob, deterministic=deterministic), int(seed) + i
        )
        rewards.append(total)
        sfs.append(sf)
        leftovers.append(leftover)
        rows.append({"name": policy.__class__.__name__, "reward": total, "shortfall_bps": sf, "leftover": leftover})
    summary = EvalSummary(
        name=policy.__class__.__name__,
        reward_mean=float(np.mean(rewards)),
        shortfall_bps_mean=float(np.mean(sfs)),
        shortfall_bps_std=float(np.std(sfs)),
        leftover_mean=float(np.mean(leftovers)),
        n=n_episodes,
    )
    return rows, summary


def _baseline_summary(
    name: BaselineId,
    n_episodes: int,
    seed: int,
    env_factory: Callable[[], OrderBookEnv] = OrderBookEnv,
) -> EvalSummary:
    from ..baselines import run_episode

    env = env_factory()
    rewards: list[float] = []
    sfs: list[float] = []
    leftovers: list[int] = []
    for i in range(n_episodes):
        res = run_episode(env, name, seed=int(seed) + i)
        rewards.append(res.reward)
        sfs.append(res.shortfall_bps)
        leftovers.append(res.leftover)
    return EvalSummary(
        name=name,
        reward_mean=float(np.mean(rewards)),
        shortfall_bps_mean=float(np.mean(sfs)),
        shortfall_bps_std=float(np.std(sfs)),
        leftover_mean=float(np.mean(leftovers)),
        n=n_episodes,
    )


def strategy_table(
    agent: Policy | None = None,
    *,
    agent_name: str = "ppo",
    n_episodes: int = 50,
    seed: int = 0,
    baselines: tuple[BaselineId, ...] = ("twap", "vwap", "pov", "passive"),
    env_factory: Callable[[], OrderBookEnv] = OrderBookEnv,
) -> list[dict]:
    """Compare agent + baselines on the same seeded episodes.

    Returns one dict per strategy:
    ``name, reward_mean, shortfall_bps_mean, shortfall_bps_std, vs_vwap_bps``
    where ``vs_vwap_bps`` is the signed *reduction* in shortfall relative to
    VWAP (positive = agent/baseline is *better* than VWAP).
    """
    rows: list[dict] = []
    vwap_sf: float | None = None
    if agent is not None:
        _, a_sum = evaluate_policy(
            agent, n_episodes=n_episodes, seed=seed,
            env_factory=env_factory, deterministic=True,
        )
        a_sum.name = agent_name
        rows.append(a_sum)
    for name in baselines:
        b = _baseline_summary(name, n_episodes, seed, env_factory=env_factory)
        if name == "vwap":
            vwap_sf = b.shortfall_bps_mean
        rows.append(b)
    out = []
    for r in rows:
        d = r.__dict__.copy()
        if vwap_sf is not None:
            d["vs_vwap_bps"] = vwap_sf - r.shortfall_bps_mean
            d["vs_vwap_pct"] = (vwap_sf - r.shortfall_bps_mean) / max(vwap_sf, 1e-9) * 100.0
        else:
            d["vs_vwap_bps"] = 0.0
            d["vs_vwap_pct"] = 0.0
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Phase 3 — per-regime, multi-seed, CI-reported evaluation (plan_2.md §6)
# ---------------------------------------------------------------------------
@dataclass
class RegimeCI:
    """Mean ± seed-family-bootstrap 95% CI for one strategy and regime."""

    name: str
    regime: str
    metric: str
    mean: float
    ci95: tuple[float, float]
    n_episodes: int
    n_seeds: int
    per_seed_mean: list[float]


def _metric_value(row: Mapping[str, object], metric: str) -> float:
    """Read one finite metric value with a useful error at the evaluation seam."""
    try:
        value = float(row[metric])
    except KeyError as exc:
        raise ValueError(f"episode row is missing metric {metric!r}") from exc
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"episode metric {metric!r} must be numeric") from exc
    if not np.isfinite(value):
        raise ValueError(f"episode metric {metric!r} must be finite")
    return value


def _integer_identifier(value: object, label: str) -> int:
    """Normalize a JSON/NumPy integer identifier without silently truncating it."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be an integer")
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if isinstance(value, str):
        return integer
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if not np.isfinite(numeric) or numeric != integer:
        raise ValueError(f"{label} must be an integer")
    return integer


def _seed_family(row: Mapping[str, object]) -> int:
    """Rows without family metadata belong to one unidentifiable cluster."""
    return _integer_identifier(row.get("seed_family", 0), "seed_family")


def _validate_equal_family_sizes(family_values: Sequence[np.ndarray]) -> None:
    """A cluster mean is episode-weighted only when every family has equal size."""
    sizes = [int(values.size) for values in family_values]
    if not sizes or any(size == 0 for size in sizes):
        raise ValueError("seed-family bootstrap needs at least one non-empty family")
    if len(sizes) < 2:
        raise ValueError("seed-family bootstrap needs at least 2 independent families")
    if len(set(sizes)) != 1:
        raise ValueError(
            "seed-family bootstrap requires equal episode counts per family; "
            f"got {sizes}"
        )


def _metric_families(
    rows: Sequence[Mapping[str, object]], metric: str,
) -> tuple[list[int], list[np.ndarray]]:
    """Group finite metric rows into complete, equally-sized seed families."""
    if len(rows) < 2:
        raise ValueError("seed-family bootstrap needs at least 2 episode rows")
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(_seed_family(row), []).append(_metric_value(row, metric))
    families = sorted(grouped)
    values = [np.asarray(grouped[family], dtype=np.float64) for family in families]
    _validate_equal_family_sizes(values)
    return families, values


def _family_bootstrap_statistics(
    family_values: Sequence[np.ndarray], *, n_boot: int, seed: int = 0x51ED,
) -> np.ndarray:
    """Resample complete seed families and return bootstrap means.

    Each draw contains one full copy of every selected family, never a moving
    block from the flattened episode stream. Equal family sizes make the
    family-sum calculation exactly the mean of those complete copies.
    """
    n_boot = _integer_identifier(n_boot, "n_boot")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    values = [np.asarray(family, dtype=np.float64) for family in family_values]
    _validate_equal_family_sizes(values)
    if not all(np.all(np.isfinite(family)) for family in values):
        raise ValueError("seed-family bootstrap needs finite metric values")

    family_size = values[0].size
    family_sums = np.asarray([family.sum() for family in values], dtype=np.float64)
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(values), size=(n_boot, len(values)))
    return family_sums[selected].sum(axis=1) / (len(values) * family_size)


def _family_bootstrap_ci(
    family_values: Sequence[np.ndarray], *, estimate: float, n_boot: int,
) -> tuple[float, float]:
    """Deterministic percentile CI, bounded to include its observed estimate."""
    stats = _family_bootstrap_statistics(family_values, n_boot=n_boot)
    if not np.all(np.isfinite(stats)):
        raise ValueError("seed-family bootstrap produced a non-finite statistic")
    lo, hi = (float(v) for v in np.quantile(stats, (0.025, 0.975)))
    lo, hi = min(lo, estimate), max(hi, estimate)
    if not np.isfinite(estimate) or not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("seed-family bootstrap CI must be finite")
    return min(lo, hi), max(lo, hi)


def _metric_direction(metric: str) -> MetricDirection:
    """Return the comparison direction rather than assuming every metric is loss."""
    try:
        return _METRIC_DIRECTIONS[metric]
    except KeyError as exc:
        supported = ", ".join(sorted(_METRIC_DIRECTIONS))
        raise ValueError(
            f"paired comparison has no direction for metric {metric!r}; "
            f"supported metrics: {supported}"
        ) from exc


def _paired_metric_families(
    agent_rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
    *,
    metric: str,
    direction: MetricDirection,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Align rows by ``(seed_family, seed)`` before making paired differences."""
    def keyed(rows: Sequence[Mapping[str, object]], label: str) -> dict[tuple[int, int], Mapping[str, object]]:
        out: dict[tuple[int, int], Mapping[str, object]] = {}
        for row in rows:
            try:
                seed = _integer_identifier(row["seed"], "seed")
            except KeyError as exc:
                raise ValueError(f"paired comparison needs a seed on every {label} row") from exc
            key = (_seed_family(row), seed)
            if key in out:
                raise ValueError("paired comparison needs unique (seed_family, seed) rows")
            out[key] = row
        return out

    agent_by_key = keyed(agent_rows, "agent")
    baseline_by_key = keyed(baseline_rows, "baseline")
    if agent_by_key.keys() != baseline_by_key.keys():
        raise ValueError(
            "paired comparison needs identical (seed_family, seed) rows for both strategies"
        )
    if len(agent_by_key) < 2:
        raise ValueError("paired comparison needs at least 2 episode rows")

    grouped: dict[int, list[float]] = {}
    baseline_values: list[float] = []
    for family, seed in sorted(agent_by_key):
        agent_value = _metric_value(agent_by_key[(family, seed)], metric)
        baseline_value = _metric_value(baseline_by_key[(family, seed)], metric)
        delta = baseline_value - agent_value if direction == "lower" else agent_value - baseline_value
        grouped.setdefault(family, []).append(delta)
        baseline_values.append(baseline_value)
    families = [np.asarray(grouped[family], dtype=np.float64) for family in sorted(grouped)]
    _validate_equal_family_sizes(families)
    return families, np.asarray(baseline_values, dtype=np.float64)


def _percent_vs_baseline(delta_mean: float, baseline_values: np.ndarray) -> float | None:
    """Relative improvement when its denominator has a meaningful positive scale.

    ``None`` deliberately replaces the old NaN for a zero, negative, or
    near-zero baseline: an unbounded percentage would fabricate a superiority
    claim. Consumers that serialized the former float must treat this one field
    as optional.
    """
    baseline_mean = float(np.mean(baseline_values))
    if baseline_mean <= 1e-12:
        return None
    pct = delta_mean / baseline_mean * 100.0
    return float(pct) if np.isfinite(pct) else None


def _episode_rows(
    env: OrderBookEnv,
    act_fn: Callable[[OrderBookEnv, np.ndarray], float],
    seeds: Sequence[int],
) -> list[dict]:
    """Full episodes for every seed; one row per episode with the honest metric set."""
    from ..execution.metrics import (
        fill_rate,
        implementation_shortfall,
        max_drawdown,
        vwap_slippage,
    )

    rows: list[dict] = []
    for sd in seeds:
        obs, _ = env.reset(seed=int(sd))
        obs = np.asarray(obs, dtype=np.float64)
        total = 0.0
        mtm_path: list[float] = [0.0]
        while True:
            a = act_fn(env, obs)
            obs, r, term, trunc, info = env.step(a)
            obs = np.asarray(obs, dtype=np.float64)
            total += float(r)
            mtm_path.append(float(info.get("pnl_ticks", 0.0)))
            if term or trunc:
                break
        market_vwap = float(info.get("market_vwap", 0.0) or 0.0)
        rows.append(
            {
                "seed": int(sd),
                "reward": total,
                "shortfall_bps": float(info["shortfall_bps"]),
                "is_bps": implementation_shortfall(env.fills, env.arrival_mid, side=1),
                "vwap_slip_bps": vwap_slippage(env.fills, market_vwap, side=1),
                "leftover": int(env.inventory),
                "completion": 1.0 - env.inventory / max(1, env.inventory0),
                "fill_rate": fill_rate(env.fills, env.inventory0),
                "mdd_ticks": max_drawdown(mtm_path),
            }
        )
    return rows


def _policy_act(policy: Policy, deterministic: bool) -> Callable[[OrderBookEnv, np.ndarray], float]:
    return lambda _env, ob: policy.act(ob, deterministic=deterministic)


def _baseline_act(name: str) -> Callable[[OrderBookEnv, np.ndarray], float]:
    from ..baselines import policy_action

    return lambda env, _ob: policy_action(name, env)  # type: ignore[arg-type]


def run_regime_episodes(
    policy: Policy | None,
    regimes: Mapping[str, Callable[[], OrderBookEnv]],
    *,
    seeds: int = 5,
    episodes_per_seed: int = 20,
    seed0: int = 0x5EED,
    baselines: Sequence[str] = (),
    agent_name: str = "ppo",
    deterministic: bool = True,
) -> dict[str, dict[str, list[dict]]]:
    """Run every strategy over paired seeded scenarios in every regime.

    Seed family ``k`` (``k < seeds``) covers episode seeds ``seed0 + k·10_000 +
    i`` for ``i < episodes_per_seed``; the agent and baselines share exogenous
    arrival draws, not their endogenous realized prices/fills.
    Returns ``{regime: {strategy: [row, ...]}}`` with
    one row per episode (``seed, reward, shortfall_bps, is_bps, vwap_slip_bps,
    leftover, completion, fill_rate, mdd_ticks``), ordered by seed family then
    episode.
    """
    strategies: list[tuple[str, Callable[[OrderBookEnv, np.ndarray], float]]] = []
    if policy is not None:
        strategies.append((agent_name, _policy_act(policy, deterministic)))
    for b in baselines:
        strategies.append((str(b), _baseline_act(str(b))))
    if not strategies:
        raise ValueError("need a policy and/or at least one baseline")
    seed_families = [
        [int(seed0) + k * 10_000 + i for i in range(episodes_per_seed)] for k in range(int(seeds))
    ]
    out: dict[str, dict[str, list[dict]]] = {}
    for regime, factory in regimes.items():
        out[regime] = {}
        for name, act in strategies:
            env = factory()
            rows: list[dict] = []
            for k, fam in enumerate(seed_families):
                for r in _episode_rows(env, act, fam):
                    r["seed_family"] = k
                    rows.append(r)
            out[regime][name] = rows
    return out


def ci_from_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    metric: str,
    name: str,
    regime: str,
    n_boot: int = 2000,
) -> RegimeCI:
    """Mean ± seed-family-bootstrap 95% CI of ``metric`` over episode rows.

    Entire seed families are sampled with replacement, so a bootstrap draw
    cannot cut through a family as a moving block over flattened rows could.
    At least two independent families with equal episode counts are required;
    one family cannot identify between-family uncertainty. Rows without
    ``seed_family`` remain one family, not independent episode observations.
    """
    fams, family_values = _metric_families(rows, metric)
    vals = np.concatenate(family_values)
    mean = float(np.mean(vals))
    lo, hi = _family_bootstrap_ci(family_values, estimate=mean, n_boot=n_boot)
    return RegimeCI(
        name=name, regime=regime, metric=metric, mean=mean,
        ci95=(lo, hi), n_episodes=len(vals), n_seeds=len(fams),
        per_seed_mean=[float(np.mean(values)) for values in family_values],
    )


def evaluate_regime_ci(
    policy: Policy | None,
    regimes: Mapping[str, Callable[[], OrderBookEnv]],
    *,
    seeds: int = 5,
    episodes_per_seed: int = 20,
    seed0: int = 0x5EED,
    metric: str = "shortfall_bps",
    baselines: Sequence[str] = (),
    agent_name: str = "ppo",
    n_boot: int = 2000,
    deterministic: bool = True,
) -> dict[str, dict[str, RegimeCI]]:
    """Per-regime mean ± 95% seed-family-bootstrap CI of ``metric`` (plan_2.md §6 items 1, 2, 5).

    ``regimes`` maps a label to an env factory (see ``envs.regimes``). Every
    strategy runs the **same** seeded episodes (``run_regime_episodes``); the
    CI resamples complete seed families rather than moving blocks over a
    flattened episode stream (``ci_from_rows``). Returns
    ``{regime: {strategy: RegimeCI}}``.

    ``metric`` may be ``shortfall_bps`` (vs arrival, positive = cost),
    ``is_bps`` (same math via ``execution.metrics``), ``vwap_slip_bps`` (vs
    **market** VWAP), ``reward``, ``leftover``, ``completion``, ``fill_rate``
    (higher is better), or ``mdd_ticks`` (lower is better).
    """
    episodes = run_regime_episodes(
        policy, regimes, seeds=seeds, episodes_per_seed=episodes_per_seed, seed0=seed0,
        baselines=baselines, agent_name=agent_name, deterministic=deterministic,
    )
    return {
        regime: {
            name: ci_from_rows(rows, metric=metric, name=name, regime=regime, n_boot=n_boot)
            for name, rows in strategies.items()
        }
        for regime, strategies in episodes.items()
    }


def paired_difference_ci(
    policy: Policy,
    baseline: str,
    regimes: Mapping[str, Callable[[], OrderBookEnv]],
    *,
    seeds: int = 5,
    episodes_per_seed: int = 20,
    seed0: int = 0x5EED,
    metric: str = "shortfall_bps",
    n_boot: int = 2000,
    episodes: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]] | None = None,
    agent_name: str = "ppo",
) -> dict[str, dict[str, float | None]]:
    """Per-regime CI of the **paired** per-episode difference ``baseline − agent``.

    Positive = the agent is better: lower for execution costs / drawdown /
    leftovers, higher for reward / completion / ``fill_rate``. Rows are aligned
    by both ``seed_family`` and ``seed`` before their difference is formed, and
    bootstrap draws resample complete paired families. This is the number a
    README line may quote: it uses identical seeded episodes for both sides, so
    tape noise cancels and the CI reflects policy differences only. Pass
    ``episodes`` (from ``run_regime_episodes``, containing both ``agent_name``
    and ``baseline``) to avoid re-running.

    ``pct_vs_baseline`` is ``None`` rather than NaN when the baseline mean is
    zero, negative, or near zero: a relative percentage has no meaningful
    denominator in that case. Consumers must treat this formerly numeric field
    as optional; positive, well-scaled baselines retain the prior calculation.
    Returns ``{regime: {"mean", "lo", "hi", "n", "frac_agent_better",
    "pct_vs_baseline"}}``.
    """
    if episodes is None:
        episodes = run_regime_episodes(
            policy, regimes, seeds=seeds, episodes_per_seed=episodes_per_seed, seed0=seed0,
            baselines=(baseline,), agent_name=agent_name,
        )
    direction = _metric_direction(metric)
    out: dict[str, dict[str, float | None]] = {}
    for regime in regimes:
        ra = episodes[regime][agent_name]
        rb = episodes[regime][baseline]
        families, baseline_values = _paired_metric_families(
            ra, rb, metric=metric, direction=direction,
        )
        diffs = np.concatenate(families)
        mean = float(np.mean(diffs))
        lo, hi = _family_bootstrap_ci(families, estimate=mean, n_boot=n_boot)
        out[regime] = {
            "mean": mean,
            "lo": lo,
            "hi": hi,
            "n": float(len(diffs)),
            "frac_agent_better": float(np.mean(diffs > 0.0)),
            "pct_vs_baseline": _percent_vs_baseline(mean, baseline_values),
        }
    return out


def format_regime_table(result: Mapping[str, Mapping[str, RegimeCI]]) -> str:
    """Monospace table: one block per regime, ``mean [lo, hi]`` per strategy."""
    lines: list[str] = []
    for regime, strategies in result.items():
        any_ci = next(iter(strategies.values()))
        lines.append(
            f"--- regime {regime} · {any_ci.metric} · {any_ci.n_seeds} seeds × "
            f"{any_ci.n_episodes // max(1, any_ci.n_seeds)} episodes ---"
        )
        lines.append(f"{'strategy':<15}{'mean':>9}{'ci95_lo':>10}{'ci95_hi':>10}{'per-seed means':>34}")
        for name, ci in strategies.items():
            seeds = " ".join(f"{v:6.2f}" for v in ci.per_seed_mean)
            lines.append(f"{name:<15}{ci.mean:>9.3f}{ci.ci95[0]:>10.3f}{ci.ci95[1]:>10.3f}  {seeds:>32}")
    return "\n".join(lines)


def format_table(rows: list[dict]) -> str:
    """Render ``strategy_table`` output as a monospace summary line per row."""
    header = f"{'strategy':<10}{'reward':>10}{'shortfall_bps':>14}{'vs_vwap%':>10}{'leftover':>10}"
    lines = [header]
    for r in rows:
        lines.append(
            f"{r['name']:<10}{r['reward_mean']:>10.2f}{r['shortfall_bps_mean']:>14.3f}"
            f"{r['vs_vwap_pct']:>9.1f}%{r['leftover_mean']:>10.2f}"
        )
    return "\n".join(lines)
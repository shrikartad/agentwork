#!/usr/bin/env python3
"""Phase 3 — fair re-verification of the PPO-vs-VWAP slippage headline (plan_2.md §6).

The Part-1 "+50.4% lower slippage than VWAP" number was produced with (1) a
regime indicator only the agent could see, (2) evaluation on the training
distribution, (3) slippage vs self-VWAP, (4) no fees / queue, (5) one seed
family and no CI, (6) a regime tuned until the number appeared. This script
re-runs the comparison with every one of those fixed:

* **Symmetric information** — two modes: ``novol`` (nobody sees the regime
  flag, obs dim 44) and ``volsym`` (the flag is in ``obs[44]`` AND every
  baseline reads the same flag via ``baselines.regime_indicator``).
* **Hold-out regimes** — the agent trains on ``highvol`` only and is scored on
  ``calm``, ``lowvol``, ``highvol_null`` (random-walk null arm), ``trending``,
  ``liquidity_shock`` as well.
* **Fees + queue ON** for every reported number (``envs.regimes.COSTS_ON``).
* **Fair baselines** — ``schedule_twap`` / ``adaptive_pov`` / ``is_aware``
  next to the legacy ``twap`` / ``vwap`` / ``pov`` / ``passive``.
* **≥5 training seeds × 5 eval seed families**, block-bootstrap 95% CIs, and
  a **paired** per-episode CI of ``best baseline − agent`` on identical tapes.
* Slippage is reported vs arrival (IS) **and** vs the tape's **market VWAP**.

The honest outcome may be "not significantly better" — that is recorded, not
hidden. Output: ``docs/results/rl_fairness.json`` + ``docs/results/rl_fairness.md``.

Run (from repo root; ~10–20 min at the defaults on one core)::

    python python_quant/scripts/rl_fairness_study.py --iters 600 --train-seeds 5
    python python_quant/scripts/rl_fairness_study.py --quick     # smoke (minutes)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1252 safe

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "python_quant"))

from nexus_quant.agents import (
    PPOConfig,
    PPOPolicy,
    ci_from_rows,
    format_regime_table,
    paired_difference_ci,
    run_regime_episodes,
    train_ppo,
)
from nexus_quant.baselines import ALL_BASELINES
from nexus_quant.envs.regimes import (
    HOLDOUT_REGIMES,
    TRAIN_REGIME,
    regime_factories,
    regime_kwargs,
)

MODES = ("novol", "volsym")
METRICS = ("shortfall_bps", "vwap_slip_bps", "completion", "fill_rate", "mdd_ticks", "reward")
PAIRED_METRICS = ("shortfall_bps", "vwap_slip_bps", "fill_rate", "mdd_ticks")
EVAL_SEED0 = 0x5EED


def _source_fingerprint() -> str:
    paths = [Path(__file__), *sorted((_ROOT / "python_quant" / "nexus_quant").rglob("*.py"))]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(_ROOT).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _runtime() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "gymnasium": version("gymnasium"), "book_backend": "StubBookAdapter"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    text = json.dumps(value, indent=1, allow_nan=False) + "\n"
    _atomic_text(path, text)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("iters", "episodes", "train_seeds", "episodes_per_seed", "n_boot"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.eval_seeds < 2:
        raise ValueError("eval_seeds must be at least 2 for seed-family confidence intervals")
    if args.episodes_per_seed > 10_000:
        raise ValueError("episodes_per_seed cannot exceed the 10,000-seed family spacing")


def _train(mode: str, seed: int, iters: int, episodes: int, out_dir: Path) -> PPOPolicy:
    path = out_dir / f"policy_{mode}_seed{seed}.npz"
    manifest_path = path.with_suffix(".json")
    cfg = PPOConfig(iterations=iters, episodes=episodes, epochs=4, eval_every=0,
                    seed=0xACE + seed, unit_seed=0x2717 + 100_000 * seed)
    context = json.loads(json.dumps({
        "schema_version": 1, "mode": mode, "training_seed": seed,
        "ppo_config": asdict(cfg),
        "env_kwargs": regime_kwargs(TRAIN_REGIME, costs=True, vol_feature=(mode == "volsym")),
        "source_fingerprint": _source_fingerprint(), "runtime": _runtime(),
    }))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (isinstance(manifest, dict) and manifest.get("context") == context
                and manifest.get("sha256") == _sha256(path)):
            policy = PPOPolicy.load(str(path))
            if policy.obs_dim == (45 if mode == "volsym" else 44) and all(
                np.isfinite(value).all() for value in policy.state_dict().values()
            ):
                print(f"  reused verified {path.name}")
                return policy
    except (OSError, ValueError, KeyError):
        pass
    factory = regime_factories((TRAIN_REGIME,), costs=True, vol_feature=(mode == "volsym"))[TRAIN_REGIME]
    t0 = time.time()
    policy, _ = train_ppo(factory, cfg)
    if not all(np.isfinite(value).all() for value in policy.state_dict().values()):
        raise ValueError("training produced non-finite policy parameters")
    out_dir.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=out_dir)
    os.close(fd)
    try:
        policy.save(temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    _atomic_json(manifest_path, {"context": context, "sha256": _sha256(path)})
    print(f"  trained {mode} seed {seed}: {iters} iters in {time.time() - t0:.0f}s -> {path.name}")
    return policy


def _ci_dict(ci) -> dict:
    return {"mean": ci.mean, "lo": ci.ci95[0], "hi": ci.ci95[1], "n": ci.n_episodes,
            "per_seed_mean": ci.per_seed_mean}


def run(args: argparse.Namespace) -> dict:
    _validate_args(args)
    out_dir = Path(args.artifacts)
    out_dir.mkdir(parents=True, exist_ok=True)
    regimes_all = (TRAIN_REGIME, *HOLDOUT_REGIMES)
    result: dict = {
        "config": {
            "iters": args.iters, "episodes": args.episodes, "train_seeds": args.train_seeds,
            "eval_seeds": args.eval_seeds, "episodes_per_seed": args.episodes_per_seed,
            "train_regime": TRAIN_REGIME, "holdout_regimes": list(HOLDOUT_REGIMES),
            "costs_on": True, "baselines": list(ALL_BASELINES),
            "n_boot": args.n_boot, "eval_seed0": EVAL_SEED0,
            "ci_method": "whole_seed_family_percentile_bootstrap",
            "source_fingerprint": _source_fingerprint(), "runtime": _runtime(),
        },
        "provenance": {},
        "modes": {},
    }
    for mode in MODES:
        print(f"\n=== mode {mode} ({'agent + baselines see the regime flag' if mode == 'volsym' else 'nobody sees a regime flag'}) ===")
        vol = mode == "volsym"
        factories = regime_factories(regimes_all, costs=True, vol_feature=vol)
        policies = [_train(mode, s, args.iters, args.episodes, out_dir) for s in range(args.train_seeds)]
        common = {"seeds": args.eval_seeds, "episodes_per_seed": args.episodes_per_seed,
                  "seed0": EVAL_SEED0}
        # every strategy runs the SAME seeded episodes once; all metrics derive from those rows
        base_eps = run_regime_episodes(None, factories, baselines=ALL_BASELINES, **common)
        agent_eps = [run_regime_episodes(pol, factories, agent_name="ppo", **common) for pol in policies]
        rows_path = out_dir / f"episodes_{mode}.json"
        _atomic_json(rows_path, {"config": result["config"], "mode": mode,
                                 "baselines": base_eps, "ppo_seeds": agent_eps})
        result["provenance"][mode] = {
            "episode_rows_sha256": _sha256(rows_path),
            "policies": [json.loads((out_dir / f"policy_{mode}_seed{s}.json").read_text(encoding="utf-8"))
                         for s in range(args.train_seeds)],
        }
        mode_res: dict = {"per_metric": {}, "paired": {}, "paired_metrics": {}}
        for metric in METRICS:
            mode_res["per_metric"][metric] = {}
            for r in regimes_all:
                seeds_ci = [_ci_dict(ci_from_rows(ae[r]["ppo"], metric=metric, name="ppo", regime=r,
                                                  n_boot=args.n_boot)) for ae in agent_eps]
                mode_res["per_metric"][metric][r] = {
                    "ppo_seeds": seeds_ci,
                    "ppo_pooled_mean": float(np.mean([d["mean"] for d in seeds_ci])),
                    "ppo_seed_std": float(np.std([d["mean"] for d in seeds_ci])),
                    "baselines": {
                        b: _ci_dict(ci_from_rows(base_eps[r][b], metric=metric, name=b, regime=r, n_boot=args.n_boot))
                        for b in ALL_BASELINES
                    },
                }
        table = {
            r: {b: ci_from_rows(base_eps[r][b], metric="shortfall_bps", name=b, regime=r, n_boot=args.n_boot)
                for b in ALL_BASELINES}
            for r in regimes_all
        }
        print(format_regime_table(table))
        # paired per-episode difference vs the best baseline and vs VWAP, per training seed
        for r in regimes_all:
            bl = mode_res["per_metric"]["shortfall_bps"][r]["baselines"]
            best = min(bl, key=lambda b: bl[b]["mean"])
            for metric in PAIRED_METRICS:
                metric_bl = mode_res["per_metric"][metric][r]["baselines"]
                entry = {"best_baseline": best, "best_baseline_mean": metric_bl[best]["mean"],
                         "vwap_mean": metric_bl["vwap"]["mean"], "vs_best": [], "vs_vwap": []}
                for pol, ae in zip(policies, agent_eps):
                    merged = {r: {"ppo": ae[r]["ppo"], **base_eps[r]}}
                    for key, target in (("vs_best", best), ("vs_vwap", "vwap")):
                        d = paired_difference_ci(pol, target, {r: factories[r]}, metric=metric,
                                                 n_boot=args.n_boot, episodes=merged, **common)[r]
                        entry[key].append(d)
                if metric == "shortfall_bps":
                    mode_res["paired"][r] = entry
                else:
                    mode_res["paired_metrics"].setdefault(metric, {})[r] = entry
            entry = mode_res["paired"][r]
            ppo_mean = mode_res["per_metric"]["shortfall_bps"][r]["ppo_pooled_mean"]
            sig = sum(1 for d in entry["vs_best"] if d["lo"] > 0)
            worse = sum(1 for d in entry["vs_best"] if d["hi"] < 0)
            print(f"  {r:<16} ppo {ppo_mean:6.3f} vs best baseline {best} {bl[best]['mean']:6.3f} | "
                  f"paired Δ(best−ppo) per seed: "
                  + " ".join(f"{d['mean']:+.2f}[{d['lo']:+.2f},{d['hi']:+.2f}]" for d in entry["vs_best"])
                  + f" | sig better {sig}/{len(policies)}, sig worse {worse}/{len(policies)}")
        result["modes"][mode] = mode_res
    return result


def render_markdown(res: dict) -> str:
    cfg = res["config"]
    lines = [
        "# RL fairness re-verification (Phase 3, plan_2.md §6)",
        "",
        (
            f"Training regime **{cfg['train_regime']}**, hold-outs {', '.join(cfg['holdout_regimes'])}. "
            f"{cfg['train_seeds']} training seeds × {cfg['eval_seeds']} eval seed families × "
            f"{cfg['episodes_per_seed']} episodes; PPO {cfg['iters']} iters × {cfg['episodes']} episodes/iter. "
            "Fees + queue model **on** for every number. Slippage positive = a cost (bps)."
        ),
        "",
        (
            "`Δ` is the **paired** per-episode difference `baseline − PPO` on identical seeded tapes "
            "(positive = PPO better) with a whole-seed-family bootstrap 95% CI; `sig` counts training seeds whose CI "
            "excludes 0 in PPO's favour / against it."
        ),
        "",
    ]
    lines += [
        ("Fill rate is the parent-order filled fraction (including terminal liquidation), not child-order fill probability. "
        "Drawdown is the maximum loss from a prior peak of aggregate inventory PnL, including the initial zero. "
        "Its historical `mdd_ticks` name denotes tick-valued PnL (ticks × shares), not a per-share price drawdown."),
        "",
        ("The best baseline is selected by mean shortfall on these evaluation episodes. Its paired CIs are "
        "conditional on that selection, unadjusted for baseline selection or multiple comparisons; "
        "counts of significant training seeds are descriptive, not independent replications."),
        "",
    ]
    for mode, mr in res["modes"].items():
        title = ("nobody observes the regime flag (obs dim 44)" if mode == "novol"
                 else "regime flag visible to the agent AND every baseline (symmetric)")
        lines += [f"## Mode `{mode}` — {title}", ""]
        lines += ["| regime | PPO shortfall (seed-mean ± seed-std) | best baseline | Δ vs best [95% CI] per training seed | sig better / worse | Δ vs VWAP (mean over seeds) | PPO completion | PPO fill rate | PPO max DD (ticks) | PPO slip vs market VWAP |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r, pm in mr["per_metric"]["shortfall_bps"].items():
            pr = mr["paired"][r]
            deltas = " ".join(f"{d['mean']:+.2f} [{d['lo']:+.2f},{d['hi']:+.2f}]" for d in pr["vs_best"])
            sig_b = sum(1 for d in pr["vs_best"] if d["lo"] > 0)
            sig_w = sum(1 for d in pr["vs_best"] if d["hi"] < 0)
            dv = np.mean([d["mean"] for d in pr["vs_vwap"]])
            percentages = [d["pct_vs_baseline"] for d in pr["vs_vwap"]]
            pv = (f"{np.mean(percentages):+.1f}%"
                  if percentages and all(v is not None and np.isfinite(v) for v in percentages)
                  else "n/a")
            comp = mr["per_metric"]["completion"][r]["ppo_pooled_mean"]
            fr = mr["per_metric"]["fill_rate"][r]["ppo_pooled_mean"]
            mdd = mr["per_metric"]["mdd_ticks"][r]["ppo_pooled_mean"]
            mdd_std = mr["per_metric"]["mdd_ticks"][r]["ppo_seed_std"]
            mv = mr["per_metric"]["vwap_slip_bps"][r]["ppo_pooled_mean"]
            lines.append(
                f"| {r} | {pm['ppo_pooled_mean']:.3f} ± {pm['ppo_seed_std']:.3f} | "
                f"{pr['best_baseline']} {pr['best_baseline_mean']:.3f} | {deltas} | {sig_b} / {sig_w} | "
                f"{dv:+.3f} bps ({pv}) | {comp:.3f} | {fr:.3f} | {mdd:.2f} ± {mdd_std:.2f} | {mv:+.3f} |"
            )
        lines += ["", "Baselines (shortfall_bps, mean [95% CI]):", ""]
        regimes = list(mr["per_metric"]["shortfall_bps"].keys())
        bnames = list(next(iter(mr["per_metric"]["shortfall_bps"].values()))["baselines"].keys())
        lines.append("| baseline | " + " | ".join(regimes) + " |")
        lines.append("|---|" + "---|" * len(regimes))
        for b in bnames:
            cells = []
            for r in regimes:
                d = mr["per_metric"]["shortfall_bps"][r]["baselines"][b]
                cells.append(f"{d['mean']:.3f} [{d['lo']:.3f},{d['hi']:.3f}]")
            lines.append(f"| {b} | " + " | ".join(cells) + " |")
        lines.append("")
        for metric, label in (("fill_rate", "Parent-order fill fraction"), ("mdd_ticks", "Inventory-PnL drawdown")):
            lines += [f"### {label} — whole-family 95% CIs", "",
                      ("PPO entries and paired differences follow training-seed order. The comparator remains the "
                       "shortfall-selected baseline; positive Δ means higher fill fraction or lower drawdown."), "",
                      "| regime | PPO mean [95% CI] per training seed | baseline mean [95% CI] | paired Δ [95% CI] per training seed |",
                      "|---|---|---|---|"]
            for r, pm in mr["per_metric"][metric].items():
                name = mr["paired"][r]["best_baseline"]
                baseline = pm["baselines"][name]
                cells = " ".join(f"{d['mean']:.4f} [{d['lo']:.4f},{d['hi']:.4f}]"
                                 for d in pm.get("ppo_seeds", [])) or "not computed"
                paired = mr.get("paired_metrics", {}).get(metric, {}).get(r, {}).get("vs_best", [])
                deltas = " ".join(f"{d['mean']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]" for d in paired) or "not computed"
                lines.append(f"| {r} | {cells} | {name} {baseline['mean']:.4f} "
                             f"[{baseline['lo']:.4f},{baseline['hi']:.4f}] | {deltas} |")
            lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--train-seeds", type=int, default=5)
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--episodes-per-seed", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--artifacts", default="python_quant/artifacts/fairness", help="policy cache (gitignored .npz)")
    ap.add_argument("--out", type=Path, default=_ROOT / "docs" / "results" / "rl_fairness.json")
    ap.add_argument("--quick", action="store_true", help="smoke settings (2 seeds, 40 iters, 4 episodes)")
    args = ap.parse_args(argv)
    if args.quick:
        args.iters, args.train_seeds, args.eval_seeds, args.episodes_per_seed, args.n_boot = 40, 2, 2, 4, 200
        args.artifacts = str(Path(args.artifacts) / "quick")
    try:
        _validate_args(args)
    except ValueError as exc:
        ap.error(str(exc))
    t0 = time.time()
    res = run(args)
    res["config"]["wall_seconds"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.out, res)
    md = args.out.with_suffix(".md")
    _atomic_text(md, render_markdown(res))
    print(f"\nwrote {args.out} and {md} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

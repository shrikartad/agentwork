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
import importlib.metadata
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
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
)

MODES = ("novol", "volsym")
METRICS = ("shortfall_bps", "vwap_slip_bps", "completion", "fill_rate", "mdd_ticks", "reward")
PPO_EPOCHS = 4
EVAL_SEED0 = 0x5EED
POLICY_CACHE_SCHEMA = 1
E7_SOURCE_FILES = (
    "python_quant/scripts/rl_fairness_study.py",
    "python_quant/nexus_quant/agents/evaluate.py",
    "python_quant/nexus_quant/agents/mlp.py",
    "python_quant/nexus_quant/agents/ppo.py",
    "python_quant/nexus_quant/baselines.py",
    "python_quant/nexus_quant/envs/order_book_env.py",
    "python_quant/nexus_quant/envs/regimes.py",
    "python_quant/nexus_quant/execution/metrics.py",
)
LIMITATIONS = (
    "This is a synthetic OrderBookEnv study, not a real NASDAQ ITCH execution result.",
    "PPO trains only on highvol; the other five regimes are simulated hold-outs.",
    (
        "The best baseline is selected separately per regime from these same evaluation families; "
        "the paired CI does not adjust for that winner selection."
    ),
    "These results do not revive the retired +50.4% exploratory claim or establish real-tape superiority.",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_fingerprint() -> dict[str, object]:
    """Fingerprint the E7 code that determines training and reported metrics."""
    digest = hashlib.sha256()
    files: dict[str, str] = {}
    for relative in E7_SOURCE_FILES:
        path = _ROOT / relative
        content = path.read_bytes()
        files[relative] = hashlib.sha256(content).hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return {"algorithm": "sha256", "combined": digest.hexdigest(), "files": files}


def _git_head() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _seed_schedule(args: argparse.Namespace) -> dict[str, object]:
    training = [
        {
            "policy_seed": seed,
            "ppo_rng_seed": 0xACE + seed,
            "ppo_unit_seed": 0x2717 + 100_000 * seed,
        }
        for seed in range(args.train_seeds)
    ]
    families = [
        {
            "seed_family": family,
            "episode_seeds": [
                EVAL_SEED0 + family * 10_000 + episode
                for episode in range(args.episodes_per_seed)
            ],
        }
        for family in range(args.eval_seeds)
    ]
    return {
        "training": {
            "per_mode": training,
            "iterations": args.iters,
            "episodes_per_iteration": args.episodes,
            "epochs": PPO_EPOCHS,
        },
        "evaluation": {
            "seed0": EVAL_SEED0,
            "family_stride": 10_000,
            "families": families,
            "episodes_per_family": args.episodes_per_seed,
            "deterministic_policy_actions": True,
        },
    }


def _cache_signature(args: argparse.Namespace, source: Mapping[str, object]) -> dict[str, object]:
    """Parameters and source state that must match before a policy cache is reused."""
    return {
        "schema": POLICY_CACHE_SCHEMA,
        "modes": list(MODES),
        "train_regime": TRAIN_REGIME,
        "costs_on": True,
        "iters": args.iters,
        "episodes_per_iteration": args.episodes,
        "epochs": PPO_EPOCHS,
        "training_seed_schedule": _seed_schedule(args)["training"],
        "source_sha256": source["combined"],
    }


def _prepare_policy_cache(
    out_dir: Path, args: argparse.Namespace, source: Mapping[str, object],
) -> dict[str, object]:
    """Create or validate an ignored cache manifest before loading saved policies."""
    manifest_path = out_dir / "fairness_policy_cache_manifest.npz"
    signature = _cache_signature(args, source)
    serialized = json.dumps(signature, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if manifest_path.exists():
        try:
            with np.load(manifest_path, allow_pickle=False) as saved:
                existing = str(saved["signature"].item())
        except (KeyError, OSError, ValueError) as exc:
            raise ValueError(f"invalid policy-cache manifest: {manifest_path}") from exc
        if existing != serialized:
            raise ValueError(
                "policy cache does not match this full training configuration/source fingerprint; "
                "choose a fresh --artifacts namespace"
            )
        state = "resumed"
    else:
        existing_policies = [
            *out_dir.glob("policy_novol_seed*.npz"),
            *out_dir.glob("policy_volsym_seed*.npz"),
        ]
        if existing_policies:
            raise ValueError(
                "policy cache has saved policies but no matching manifest; choose a fresh --artifacts namespace"
            )
        np.savez_compressed(manifest_path, signature=np.asarray(serialized))
        state = "fresh"
    return {
        "namespace": str(out_dir),
        "manifest": manifest_path.name,
        "state": state,
        "signature": signature,
    }


def _policy_metadata(path: Path, *, state: str, seconds: float) -> dict[str, object]:
    return {
        "file": path.name,
        "cache_status": state,
        "wall_seconds": round(seconds, 3),
        "sha256": _sha256(path),
    }


def _run_provenance(
    args: argparse.Namespace, source: Mapping[str, object], started_utc: str,
) -> dict[str, object]:
    thread_limits = {
        name: os.environ.get(name)
        for name in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        )
        if os.environ.get(name) is not None
    }
    return {
        "started_utc": started_utc,
        "source": {"git_head": _git_head(), "fingerprint": source},
        "command": {
            "argv": list(sys.argv),
            "python_executable": sys.executable,
            "working_directory": str(Path.cwd()),
            "thread_limits": thread_limits,
        },
        "seed_schedule": _seed_schedule(args),
        "environment": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "gymnasium": importlib.metadata.version("gymnasium"),
        },
        "limitations": list(LIMITATIONS),
    }


def _train(
    mode: str, seed: int, iters: int, episodes: int, out_dir: Path,
) -> tuple[PPOPolicy, dict[str, object]]:
    path = out_dir / f"policy_{mode}_seed{seed}.npz"
    t0 = time.time()
    if path.exists():
        policy = PPOPolicy.load(str(path))
        metadata = _policy_metadata(path, state="cached", seconds=time.time() - t0)
        print(f"  reused {mode} seed {seed}: {path.name}")
        return policy, metadata
    factory = regime_factories((TRAIN_REGIME,), costs=True, vol_feature=(mode == "volsym"))[TRAIN_REGIME]
    cfg = PPOConfig(iterations=iters, episodes=episodes, epochs=PPO_EPOCHS, eval_every=0,
                    seed=0xACE + seed, unit_seed=0x2717 + 100_000 * seed)
    policy, _ = train_ppo(factory, cfg)
    policy.save(str(path))
    seconds = time.time() - t0
    print(f"  trained {mode} seed {seed}: {iters} iters in {seconds:.0f}s -> {path.name}")
    return policy, _policy_metadata(path, state="trained", seconds=seconds)


def _ci_dict(ci) -> dict:
    return {"mean": ci.mean, "lo": ci.ci95[0], "hi": ci.ci95[1], "n": ci.n_episodes,
            "per_seed_mean": ci.per_seed_mean}


def run(args: argparse.Namespace, *, source: Mapping[str, object] | None = None) -> dict:
    out_dir = Path(args.artifacts)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = _source_fingerprint() if source is None else source
    policy_cache = _prepare_policy_cache(out_dir, args, source)
    regimes_all = (TRAIN_REGIME, *HOLDOUT_REGIMES)
    result: dict = {
        "config": {
            "iters": args.iters, "episodes": args.episodes, "train_seeds": args.train_seeds,
            "eval_seeds": args.eval_seeds, "episodes_per_seed": args.episodes_per_seed,
            "train_regime": TRAIN_REGIME, "holdout_regimes": list(HOLDOUT_REGIMES),
            "costs_on": True, "baselines": list(ALL_BASELINES), "modes": list(MODES),
            "bootstrap_replicates": args.n_boot,
        },
        "provenance": {"policy_cache": policy_cache, "training": {}},
        "modes": {},
    }
    for mode in MODES:
        print(f"\n=== mode {mode} ({'agent + baselines see the regime flag' if mode == 'volsym' else 'nobody sees a regime flag'}) ===")
        vol = mode == "volsym"
        factories = regime_factories(regimes_all, costs=True, vol_feature=vol)
        trained = [_train(mode, s, args.iters, args.episodes, out_dir) for s in range(args.train_seeds)]
        policies = [policy for policy, _ in trained]
        result["provenance"]["training"][mode] = [metadata for _, metadata in trained]
        common = {
            "seeds": args.eval_seeds,
            "episodes_per_seed": args.episodes_per_seed,
            "seed0": EVAL_SEED0,
        }
        # every strategy runs the SAME seeded episodes once; all metrics derive from those rows
        base_eps = run_regime_episodes(None, factories, baselines=ALL_BASELINES, **common)
        agent_eps = [run_regime_episodes(pol, factories, agent_name="ppo", **common) for pol in policies]
        mode_res: dict = {"per_metric": {}, "paired": {}}
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
            entry = {"best_baseline": best, "best_baseline_mean": bl[best]["mean"],
                     "vwap_mean": bl["vwap"]["mean"], "vs_best": [], "vs_vwap": []}
            for pol, ae in zip(policies, agent_eps):
                merged = {r: {"ppo": ae[r]["ppo"], **base_eps[r]}}
                for key, target in (("vs_best", best), ("vs_vwap", "vwap")):
                    d = paired_difference_ci(pol, target, {r: factories[r]}, metric="shortfall_bps",
                                             n_boot=args.n_boot, episodes=merged, **common)[r]
                    entry[key].append(d)
            mode_res["paired"][r] = entry
            ppo_mean = mode_res["per_metric"]["shortfall_bps"][r]["ppo_pooled_mean"]
            sig = sum(1 for d in entry["vs_best"] if d["lo"] > 0)
            worse = sum(1 for d in entry["vs_best"] if d["hi"] < 0)
            print(f"  {r:<16} ppo {ppo_mean:6.3f} vs best baseline {best} {bl[best]['mean']:6.3f} | "
                  f"paired Δ(best−ppo) per seed: "
                  + " ".join(f"{d['mean']:+.2f}[{d['lo']:+.2f},{d['hi']:+.2f}]" for d in entry["vs_best"])
                  + f" | sig better {sig}/{len(policies)}, sig worse {worse}/{len(policies)}")
        result["modes"][mode] = mode_res
    return result


def _ppo_metric_cell(summary: Mapping[str, object], *, decimals: int) -> str:
    """Render the across-policy mean and the envelope of per-policy family CIs."""
    mean = float(summary["ppo_pooled_mean"])
    std = float(summary["ppo_seed_std"])
    per_policy = summary.get("ppo_seeds", [])
    bounds: list[tuple[float, float]] = []
    if isinstance(per_policy, list):
        for ci in per_policy:
            if isinstance(ci, Mapping):
                lo, hi = float(ci["lo"]), float(ci["hi"])
                if math.isfinite(lo) and math.isfinite(hi):
                    bounds.append((lo, hi))
    rendered = f"{mean:.{decimals}f} ± {std:.{decimals}f}"
    if bounds:
        rendered += f" [{min(lo for lo, _ in bounds):.{decimals}f},{max(hi for _, hi in bounds):.{decimals}f}]"
    return rendered


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
        (
            "The best baseline is selected separately per regime by its smallest baseline mean on these same "
            "evaluation families. Therefore the paired `Δ vs best` intervals are descriptive and do not adjust "
            "for winner selection. This synthetic study does not revive the retired +50.4% exploratory claim."
        ),
        "",
    ]
    provenance = res.get("provenance")
    if isinstance(provenance, Mapping):
        source = provenance.get("source", {})
        command = provenance.get("command", {})
        seeds = provenance.get("seed_schedule", {})
        environment = provenance.get("environment", {})
        runtime = provenance.get("runtime", {})
        cache = provenance.get("policy_cache", {})
        if not isinstance(source, Mapping):
            source = {}
        if not isinstance(command, Mapping):
            command = {}
        if not isinstance(seeds, Mapping):
            seeds = {}
        if not isinstance(environment, Mapping):
            environment = {}
        if not isinstance(runtime, Mapping):
            runtime = {}
        if not isinstance(cache, Mapping):
            cache = {}
        fingerprint = source.get("fingerprint", {})
        if not isinstance(fingerprint, Mapping):
            fingerprint = {}
        argv = command.get("argv", [])
        if not isinstance(argv, list):
            argv = []
        exact_command = shlex.join([str(command.get("python_executable", "python")), *map(str, argv)])
        training = seeds.get("training", {})
        evaluation = seeds.get("evaluation", {})
        if not isinstance(training, Mapping):
            training = {}
        if not isinstance(evaluation, Mapping):
            evaluation = {}
        family_count = len(evaluation.get("families", [])) if isinstance(evaluation.get("families"), list) else 0
        thread_limits = command.get("thread_limits", {})
        if not isinstance(thread_limits, Mapping):
            thread_limits = {}
        limits = ", ".join(f"{key}={value}" for key, value in thread_limits.items()) or "not recorded"
        lines += [
            "## Reproducibility",
            "",
            f"- Git HEAD at launch: `{source.get('git_head') or 'unavailable'}`.",
            f"- E7 source SHA-256: `{fingerprint.get('combined', 'unavailable')}`.",
            f"- Exact command: `{exact_command}`.",
            (
                f"- Training: {cfg['train_seeds']} policies per mode; {training.get('iterations', cfg['iters'])} "
                f"iterations × {training.get('episodes_per_iteration', cfg['episodes'])} episodes/iteration × "
                f"{training.get('epochs', PPO_EPOCHS)} epochs."
            ),
            (
                f"- Evaluation: {family_count} seed families × "
                f"{evaluation.get('episodes_per_family', cfg['episodes_per_seed'])} episodes, "
                f"seed0 `{evaluation.get('seed0', EVAL_SEED0)}`, family stride "
                f"`{evaluation.get('family_stride', 10_000)}`; exact seed lists are in the JSON."
            ),
            f"- Bootstrap: {cfg.get('bootstrap_replicates', 'unrecorded')} whole-family replicates per CI.",
            (
                f"- Runtime: {runtime.get('wall_seconds', cfg.get('wall_seconds', 'unrecorded'))} s "
                f"({runtime.get('started_utc', provenance.get('started_utc', 'unrecorded'))} to "
                f"{runtime.get('completed_utc', 'unrecorded')})."
            ),
            (
                f"- Environment: Python {environment.get('python', 'unrecorded')}; NumPy "
                f"{environment.get('numpy', 'unrecorded')}; Gymnasium "
                f"{environment.get('gymnasium', 'unrecorded')}; thread limits {limits}."
            ),
            (
                f"- Policy cache: `{cache.get('namespace', 'unrecorded')}` "
                f"({cache.get('state', 'unrecorded')}, manifest `{cache.get('manifest', 'unrecorded')}`)."
            ),
            "",
            "## Scope and limitations",
            "",
        ]
        lines.extend(f"- {limitation}" for limitation in provenance.get("limitations", []))
        lines.append("")
    for mode, mr in res["modes"].items():
        title = ("nobody observes the regime flag (obs dim 44)" if mode == "novol"
                 else "regime flag visible to the agent AND every baseline (symmetric)")
        lines += [f"## Mode `{mode}` — {title}", ""]
        lines += ["| regime | PPO shortfall (seed-mean ± seed-std) | best baseline | Δ vs best [95% CI] per training seed | sig better / worse | Δ vs VWAP (mean over seeds) | PPO completion† | PPO fill rate† | PPO max DD (ticks)† | PPO slip vs market VWAP |",
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
            comp = _ppo_metric_cell(mr["per_metric"]["completion"][r], decimals=3)
            fr = _ppo_metric_cell(mr["per_metric"]["fill_rate"][r], decimals=3)
            mdd = _ppo_metric_cell(mr["per_metric"]["mdd_ticks"][r], decimals=2)
            mv = mr["per_metric"]["vwap_slip_bps"][r]["ppo_pooled_mean"]
            lines.append(
                f"| {r} | {pm['ppo_pooled_mean']:.3f} ± {pm['ppo_seed_std']:.3f} | "
                f"{pr['best_baseline']} {pr['best_baseline_mean']:.3f} | {deltas} | {sig_b} / {sig_w} | "
                f"{dv:+.3f} bps ({pv}) | {comp} | {fr} | {mdd} | {mv:+.3f} |"
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
    lines += [
        (
            f"† PPO point estimate is the mean over {cfg['train_seeds']} independently trained policies; `±` is their "
            f"standard deviation. Brackets are the envelope of the {cfg['train_seeds']} per-policy, whole-evaluation-family "
            "bootstrap 95% CIs (not a pooled joint CI); all component CIs are retained in the JSON."
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def _validate_json(value: object, path: str = "$") -> None:
    """Reject NaN/Infinity before publishing a report that claims finite CIs."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON value at {path}")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object key at {path} must be a string")
            _validate_json(nested, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _validate_json(nested, f"{path}[{index}]")
        return
    if value is None or isinstance(value, (bool, int, str)):
        return
    raise ValueError(f"unsupported JSON value at {path}: {type(value).__name__}")


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
    t0 = time.time()
    started_utc = datetime.now(timezone.utc).isoformat()
    source = _source_fingerprint()
    launch_provenance = _run_provenance(args, source, started_utc)
    res = run(args, source=source)
    wall_seconds = round(time.time() - t0, 1)
    res["config"]["wall_seconds"] = wall_seconds
    res["provenance"].update(launch_provenance)
    res["provenance"]["runtime"] = {
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": wall_seconds,
    }
    _validate_json(res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, allow_nan=False) + "\n")
    md = args.out.with_suffix(".md")
    md.write_text(render_markdown(res))
    print(f"\nwrote {args.out} and {md} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

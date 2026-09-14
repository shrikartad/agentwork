"""Full-run provenance, cache invalidation, and E7 report coverage."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from nexus_quant.agents import PPOPolicy

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rl_fairness_study.py"
_SPEC = importlib.util.spec_from_file_location("fairness_runner_test", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
study = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(study)


def _args(tmp_path, **overrides):
    values = {"iters": 1, "episodes": 1, "train_seeds": 2, "eval_seeds": 2,
              "episodes_per_seed": 1, "n_boot": 20, "artifacts": str(tmp_path)}
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.fixture
def training_calls(monkeypatch):
    calls = []

    def train(factory, config):
        calls.append(config)
        return PPOPolicy(factory().observation_space.shape[0], seed=config.seed), []

    monkeypatch.setattr(study, "train_ppo", train)
    return calls


def test_cache_reuses_only_matching_training_provenance(tmp_path, training_calls, monkeypatch):
    first = study._train("novol", 0, 1, 1, tmp_path)
    second = study._train("novol", 0, 1, 1, tmp_path)
    assert len(training_calls) == 1
    for key, value in first.state_dict().items():
        np.testing.assert_array_equal(value, second.state_dict()[key])

    study._train("novol", 0, 2, 1, tmp_path)
    study._train("novol", 0, 2, 2, tmp_path)
    monkeypatch.setattr(study, "_source_fingerprint", lambda: "changed-source")
    study._train("novol", 0, 2, 2, tmp_path)
    monkeypatch.setattr(study, "_runtime", lambda: {"numpy": "changed-version"})
    study._train("novol", 0, 2, 2, tmp_path)
    assert len(training_calls) == 5


@pytest.mark.parametrize("damage", ("missing", "invalid-json", "nonobject", "checksum"))
def test_cache_rejects_missing_or_corrupt_metadata(tmp_path, training_calls, damage):
    study._train("volsym", 1, 1, 1, tmp_path)
    manifest = tmp_path / "policy_volsym_seed1.json"
    if damage == "missing":
        manifest.unlink()
    elif damage == "invalid-json":
        manifest.write_text("{", encoding="utf-8")
    elif damage == "nonobject":
        manifest.write_text("[]", encoding="utf-8")
    else:
        (tmp_path / "policy_volsym_seed1.npz").write_bytes(b"interrupted policy")
    study._train("volsym", 1, 1, 1, tmp_path)
    assert len(training_calls) == 2


@pytest.mark.parametrize("overrides", ({"iters": 0}, {"episodes": 0}, {"train_seeds": 0},
                                     {"eval_seeds": 1}, {"episodes_per_seed": 0},
                                     {"episodes_per_seed": 10_001}, {"n_boot": 0}))
def test_invalid_protocol_fails_before_training(tmp_path, training_calls, overrides):
    with pytest.raises(ValueError):
        study.run(_args(tmp_path, **overrides))
    assert not training_calls


def test_run_preserves_rows_and_publishes_paired_e7_cis(tmp_path, training_calls):
    result = study.run(_args(tmp_path))
    assert len(training_calls) == 4
    assert result["config"]["n_boot"] == 20
    assert result["config"]["source_fingerprint"] == study._source_fingerprint()
    for mode, block in result["modes"].items():
        path = tmp_path / f"episodes_{mode}.json"
        assert result["provenance"][mode]["episode_rows_sha256"] == study._sha256(path)
        rows = json.loads(path.read_text(encoding="utf-8"))
        assert len(rows["ppo_seeds"]) == 2
        for metric in ("vwap_slip_bps", "fill_rate", "mdd_ticks"):
            for regime, comparisons in block["paired_metrics"][metric].items():
                name = comparisons["best_baseline"]
                baseline = block["per_metric"][metric][regime]["baselines"][name]["mean"]
                direction = 1 if metric == "fill_rate" else -1
                for seed, ci in enumerate(comparisons["vs_best"]):
                    agent = block["per_metric"][metric][regime]["ppo_seeds"][seed]["mean"]
                    assert ci["lo"] <= ci["mean"] <= ci["hi"]
                    assert ci["mean"] == pytest.approx(direction * (agent - baseline))
    text = study.render_markdown(result)
    assert "Parent-order fill fraction — whole-family 95% CIs" in text
    assert "Inventory-PnL drawdown — whole-family 95% CIs" in text
    assert "not computed" not in text
    json.dumps(result, allow_nan=False)
    assert study.run(_args(tmp_path)) == result
    assert len(training_calls) == 4


def test_atomic_json_keeps_prior_report_on_nonfinite_value(tmp_path):
    path = tmp_path / "report.json"
    study._atomic_json(path, {"complete": True})
    with pytest.raises(ValueError):
        study._atomic_json(path, {"mean": float("nan")})
    assert json.loads(path.read_text()) == {"complete": True}


def test_quick_cli_does_not_overwrite_full_study(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(study, "_ROOT", tmp_path)
    monkeypatch.setattr(study, "run", lambda args: (seen.append(args) or {"config": {}}))
    monkeypatch.setattr(study, "render_markdown", lambda _: "smoke study\n")
    full = tmp_path / "docs" / "results" / "rl_fairness.json"
    full.parent.mkdir(parents=True)
    full.write_text("full study", encoding="utf-8")
    assert study.main(["--quick", "--artifacts", str(tmp_path / "cache")]) == 0
    assert seen[0].episodes == 4
    assert seen[0].out == full.parent / "quick" / full.name
    assert full.read_text() == "full study"
    assert seen[0].out.is_file()

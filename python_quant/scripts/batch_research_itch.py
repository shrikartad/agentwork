#!/usr/bin/env python3
"""Run the existing real-tape E1–E6 study across public ITCH days, resumably.

This command deliberately keeps the per-day methodology in
``scripts/run_research.py``: it streams and slices one public tape at a time,
then calls that module's replay, IC, fill, and adverse-selection studies.  It
never pools event rows across days, so the cross-day output is an index of
separate time-ordered studies rather than a new performance claim.

Examples (from the repository root)::

    python python_quant/scripts/batch_research_itch.py \
        --days 12302019,01302020 --symbols AAPL,QQQ
    python python_quant/scripts/batch_research_itch.py --days all \
        --max-gz-bytes 67108864 --skip-existing

The source slices live under ``data/itch/<day>/``.  Results live under
``docs/results/multi_day/<day>/`` with a per-day ``research_manifest.json``;
``batch_manifest.json`` and ``multi_day_summary.{json,md}`` are atomically
updated after every day.  A range-limited source is always labelled a partial
GZIP prefix, even when it happened to include regular-session rows.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import fetch_itch
import run_research

SCHEMA_VERSION = 1
BATCH_MANIFEST_NAME = "batch_manifest.json"
RESEARCH_MANIFEST_NAME = "research_manifest.json"
SUMMARY_JSON_NAME = "multi_day_summary.json"
SUMMARY_MD_NAME = "multi_day_summary.md"
HORIZONS = (1, 5, 10, 25)
FILL_HORIZONS = (10, 50, 100, 500, 1000, 5000)
ADVERSE_HORIZONS = (1, 5, 25)
N_BOOT = 200  # Matches run_research.py's default; this is not tuned per day.


def parse_days(value: str) -> list[str]:
    """Validate an explicit public sample-day list, or expand ``all``.

    Restricting dates to the curated public catalogue avoids accidental large
    requests to an arbitrary URL and makes a batch reproducible from its CLI.
    """
    raw = value.strip()
    if raw.lower() == "all":
        return list(fetch_itch.PUBLIC_SAMPLE_DAYS)
    if not raw:
        raise argparse.ArgumentTypeError("--days must be a comma-separated public sample date list or 'all'")

    known = set(fetch_itch.PUBLIC_SAMPLE_DAYS) | set(fetch_itch.UNAVAILABLE_SAMPLE_DAYS)
    days: list[str] = []
    for item in raw.split(","):
        day = item.strip()
        if not re.fullmatch(r"\d{8}", day):
            raise argparse.ArgumentTypeError(f"invalid date {day!r}; expected MMDDYYYY")
        try:
            dt.date(int(day[4:]), int(day[:2]), int(day[2:4]))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid calendar date {day!r}") from exc
        if day not in known:
            raise argparse.ArgumentTypeError(
                f"{day} is not in fetch_itch.PUBLIC_SAMPLE_DAYS; use --days all or a catalogued public day"
            )
        if day not in days:
            days.append(day)
    if not days:
        raise argparse.ArgumentTypeError("--days must name at least one public sample date")
    return days


def parse_symbols(value: str) -> list[str]:
    """Normalize requested tickers and reject values that could escape a slice path."""
    symbols: list[str] = []
    for item in value.split(","):
        symbol = item.strip().upper()
        if not symbol:
            continue
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]*", symbol):
            raise argparse.ArgumentTypeError(f"invalid ticker {item!r}")
        if symbol not in symbols:
            symbols.append(symbol)
    if not symbols:
        raise argparse.ArgumentTypeError("--symbols must name at least one ticker")
    return symbols


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _json_value(value: Any) -> Any:
    """Convert NumPy scalars and non-finite study values to portable JSON."""
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    item = getattr(value, "item", None)
    if callable(item):
        return _json_value(item())
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    """Replace a result file only after its complete content reaches disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_write_json(path: Path, payload: Any) -> None:
    text = json.dumps(_json_value(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_write_text(path, text)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pipeline_fingerprint() -> str:
    """Invalidate cached analyses when their executable Python source changes."""
    root = _SCRIPT_DIR.parent
    files = [Path(__file__), _SCRIPT_DIR / "run_research.py"]
    files.extend(sorted((root / "nexus_quant").rglob("*.py")))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_fetch_manifest(
    day: str,
    symbols: list[str],
    out_dir: Path,
    *,
    max_gz_bytes: int | None,
    base: str = fetch_itch.DEFAULT_BASE,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Return a source manifest only when every requested slice is verifiable.

    An old or partial file is intentionally not reusable as a full-day input:
    the manifest must record both the requested bound and whether the GZIP
    stream reached EOF.  Slice SHA-256 values are rechecked before analysis.
    """
    errors: list[str] = []
    day_dir = out_dir / day
    manifest_path = day_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest is None:
        return None, [f"missing or invalid manifest: {manifest_path}"]

    if manifest.get("day") != day:
        errors.append(f"manifest day is {manifest.get('day')!r}, expected {day!r}")
    expected_url = fetch_itch.tape_url(day, base)
    if manifest.get("source_url") != expected_url:
        errors.append("manifest source_url does not match the requested public source")

    expected_limited = max_gz_bytes is not None
    if manifest.get("range_limited") is not expected_limited:
        errors.append("manifest range_limited does not match --max-gz-bytes")
    if "max_gz_bytes_requested" not in manifest:
        errors.append("manifest lacks max_gz_bytes_requested provenance")
    elif manifest.get("max_gz_bytes_requested") != max_gz_bytes:
        errors.append("manifest max_gz_bytes_requested does not match this request")
    if not isinstance(manifest.get("gzip_stream_complete"), bool):
        errors.append("manifest lacks gzip_stream_complete provenance")
    elif not expected_limited and manifest["gzip_stream_complete"] is not True:
        errors.append("unbounded fetch did not reach the end of the GZIP stream")

    fetched = manifest.get("gz_bytes_fetched")
    if not _int(fetched) or fetched < 0:
        errors.append("manifest gz_bytes_fetched is invalid")
    elif max_gz_bytes is not None and fetched > max_gz_bytes:
        errors.append("manifest fetched more gzip bytes than --max-gz-bytes")
    if not _is_sha256(manifest.get("gz_sha256")):
        errors.append("manifest lacks a valid fetched-gzip SHA-256")

    entries = manifest.get("symbols")
    if not isinstance(entries, dict):
        errors.append("manifest symbols is invalid")
        entries = {}
    for symbol in symbols:
        entry = entries.get(symbol)
        if not isinstance(entry, dict):
            errors.append(f"manifest has no requested {symbol} slice")
            continue
        filename = entry.get("file")
        if filename != f"{symbol}.itch":
            errors.append(f"manifest file for {symbol} is not the expected slice name")
            continue
        path = day_dir / filename
        if not path.is_file():
            errors.append(f"missing {symbol} slice: {path}")
            continue
        expected_size = entry.get("bytes")
        if not _int(expected_size) or expected_size < 0 or path.stat().st_size != expected_size:
            errors.append(f"size mismatch for {symbol} slice")
        expected_hash = entry.get("sha256")
        if not _is_sha256(expected_hash) or _sha256(path) != expected_hash:
            errors.append(f"SHA-256 mismatch for {symbol} slice")
    return (manifest if not errors else None), errors


def _coverage(manifest: dict[str, Any]) -> dict[str, Any]:
    limited = bool(manifest["range_limited"])
    return {
        "kind": "partial_gzip_prefix" if limited else "full_day",
        "is_full_day": not limited and bool(manifest["gzip_stream_complete"]),
        "range_limited": limited,
        "max_gz_bytes_requested": manifest.get("max_gz_bytes_requested"),
        "gz_bytes_fetched": manifest.get("gz_bytes_fetched"),
        "gzip_stream_complete": manifest.get("gzip_stream_complete"),
        "last_tape_ts_ns": manifest.get("last_tape_ts_ns"),
        "session_events": manifest.get("session_events", []),
    }


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def fetch_and_promote(
    day: str,
    symbols: list[str],
    out_dir: Path,
    *,
    max_gz_bytes: int | None,
    base: str,
    download_workers: int = 1,
    fetcher: Callable[..., dict[str, Any]] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Fetch to a sibling staging directory and atomically install a valid day."""
    out_dir.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f".{day}.fetch-", dir=out_dir))
    target = out_dir / day
    backup: Path | None = None
    fetch_fn = fetcher or fetch_itch.fetch
    try:
        fetch_fn(
            day=day,
            symbols=set(symbols),
            out_root=staging_root,
            base=base,
            max_gz_bytes=max_gz_bytes,
            download_workers=download_workers,
            download_cache=out_dir / ".downloads" / day,
            log=log,
        )
        manifest, errors = validate_fetch_manifest(
            day, symbols, staging_root, max_gz_bytes=max_gz_bytes, base=base
        )
        if manifest is None:
            raise RuntimeError("downloaded slices failed validation: " + "; ".join(errors))
        staged_day = staging_root / day
        if not staged_day.is_dir():
            raise RuntimeError("fetch completed without a staged day directory")

        if target.exists() or target.is_symlink():
            backup = out_dir / f".{day}.superseded-{uuid.uuid4().hex}"
            os.replace(target, backup)
        try:
            os.replace(staged_day, target)
        except BaseException:
            if backup is not None and not (target.exists() or target.is_symlink()):
                os.replace(backup, target)
            raise
        if backup is not None:
            _remove_path(backup)
            backup = None
        return manifest
    finally:
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        if staging_root.exists():
            _remove_path(staging_root)


def _research_paths(results_dir: Path, day: str) -> tuple[Path, Path, Path]:
    day_dir = results_dir / day
    return day_dir, day_dir / RESEARCH_MANIFEST_NAME, day_dir / f"real_tape_{day}.md"


def _result_payload(rep: dict[str, Any]) -> dict[str, Any]:
    """Build the same E1–E6 result shape written by run_research.py."""
    stats = rep["stats"]
    ic = run_research.ic_study(rep, horizons=HORIZONS, train=0.6, val=0.2, n_boot=N_BOOT)
    fill = run_research.fill_study(rep, horizons_events=FILL_HORIZONS)
    adverse = run_research.adverse_study(rep, horizons=ADVERSE_HORIZONS)
    return {
        "tape": {
            "messages": stats.messages,
            "events": stats.emitted,
            "truncated": stats.truncated,
            "skipped_type": stats.skipped_type,
            "regular_events": rep["n_regular"],
            "integrity_issues": rep["issues"],
            "tracker_unknown_id": rep["tracker"].stats["unknown_id"],
            "t_parse_s": rep["t_parse"],
            "t_replay_s": rep["t_replay"],
        },
        "ic": ic,
        "fill": fill,
        "adverse": adverse,
    }


def _day_markdown(day: str, per_symbol: dict[str, dict[str, Any]], coverage: dict[str, Any]) -> str:
    """Add an explicit provenance/coverage banner to the existing report renderer."""
    rendered = run_research.render_md(day, per_symbol)
    if coverage["is_full_day"]:
        note = (
            "## Coverage and provenance\n\n"
            "**Full-day source stream.** The unbounded GZIP response reached EOF before slicing. "
            "Exact source and slice hashes are recorded in `research_manifest.json`.\n"
        )
    else:
        requested = coverage.get("max_gz_bytes_requested")
        note = (
            "## Coverage and provenance\n\n"
            f"**Partial GZIP prefix — not a full-day result.** This run requested at most {requested:,} "
            "compressed bytes. It may contain no regular-session rows and must not be used as a full-tape "
            "or cross-day inference claim; exact source and slice hashes are recorded in "
            "`research_manifest.json`.\n"
        )
    lines = rendered.splitlines()
    # render_md always begins with a title and a blank line, but retain the title if that changes.
    insertion = [note.rstrip(), ""]
    return "\n".join(lines[:2] + insertion + lines[2:]) + "\n"


def run_day_research(
    day: str,
    symbols: list[str],
    out_dir: Path,
    results_dir: Path,
    source_manifest: dict[str, Any],
    *,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run the unmodified E1–E6 study functions once per requested symbol."""
    source_dir = out_dir / day
    source_manifest_path = source_dir / "manifest.json"
    day_dir, result_manifest_path, markdown_path = _research_paths(results_dir, day)
    day_dir.mkdir(parents=True, exist_ok=True)
    coverage = _coverage(source_manifest)
    source_slices = source_manifest["symbols"]
    per_symbol: dict[str, dict[str, Any]] = {}
    outputs: dict[str, dict[str, Any]] = {}

    for symbol in symbols:
        slice_path = source_dir / f"{symbol}.itch"
        log(f"== {day} {symbol}: E1–E6 ==")
        rep = run_research.replay_symbol(slice_path, log=log)
        try:
            result = _result_payload(rep)
        finally:
            # Replay data can be large.  Each day/symbol completes before the next begins.
            del rep
        result_path = day_dir / f"real_tape_{day}_{symbol}.json"
        _atomic_write_json(result_path, result)
        per_symbol[symbol] = result
        outputs[symbol] = {
            "file": result_path.name,
            "sha256": _sha256(result_path),
            "regular_events": result["tape"]["regular_events"],
            "e1_e4_rows": len(result["ic"]["rows"]),
            "e5_orders": result["fill"]["n_orders_regular"],
            "e6_passive_fills": result["adverse"].get("n_fills", 0),
        }

    _atomic_write_text(markdown_path, _day_markdown(day, per_symbol, coverage))
    regular_rows = sum(int(item["tape"]["regular_events"]) for item in per_symbol.values())
    if coverage["is_full_day"]:
        status = "completed" if regular_rows else "completed_no_regular_session_rows"
    else:
        status = "completed_partial" if regular_rows else "completed_partial_no_regular_session_rows"
    research_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "nexus-lob-itch-day-research",
        "day": day,
        "requested_symbols": symbols,
        "status": status,
        "completed_at_unix": int(time.time()),
        "coverage": coverage,
        "source": {
            "source_url": source_manifest["source_url"],
            "manifest_file": "manifest.json",
            "manifest_sha256": _sha256(source_manifest_path),
            "gz_sha256": source_manifest["gz_sha256"],
            "slices": {
                symbol: {
                    "file": source_slices[symbol]["file"],
                    "bytes": source_slices[symbol]["bytes"],
                    "sha256": source_slices[symbol]["sha256"],
                }
                for symbol in symbols
            },
        },
        "pipeline": {
            "source_script": "python_quant/scripts/run_research.py",
            "source_sha256": _pipeline_fingerprint(),
            "experiments": ["E1-E4", "E5", "E6"],
            "regular_session_only": True,
            "horizons": list(HORIZONS),
            "fill_horizons_events": list(FILL_HORIZONS),
            "adverse_horizons_events": list(ADVERSE_HORIZONS),
            "n_boot": N_BOOT,
            "pooled_cross_day_inference": False,
        },
        "symbols": outputs,
        "markdown": {"file": markdown_path.name, "sha256": _sha256(markdown_path)},
    }
    # The per-day manifest is deliberately written last: it is the completion marker.
    _atomic_write_json(result_manifest_path, research_manifest)
    return research_manifest


def validate_research_manifest(
    day: str,
    symbols: list[str],
    out_dir: Path,
    results_dir: Path,
    source_manifest: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate an analysis completion marker against the current source hashes."""
    source_path = out_dir / day / "manifest.json"
    day_dir, manifest_path, markdown_path = _research_paths(results_dir, day)
    result = _read_json(manifest_path)
    if result is None:
        return None, [f"missing or invalid research manifest: {manifest_path}"]
    errors: list[str] = []
    if result.get("schema_version") != SCHEMA_VERSION or result.get("kind") != "nexus-lob-itch-day-research":
        errors.append("research manifest schema is not recognized")
    if result.get("day") != day:
        errors.append("research manifest day does not match")
    requested = result.get("requested_symbols")
    if (
        not isinstance(requested, list)
        or not all(isinstance(symbol, str) for symbol in requested)
        or not all(symbol in requested for symbol in symbols)
    ):
        errors.append("research manifest does not cover every requested symbol")
    expected_coverage = _coverage(source_manifest)
    if result.get("coverage") != expected_coverage:
        errors.append("research coverage does not match the current source manifest")
    expected_statuses = (
        {"completed", "completed_no_regular_session_rows"}
        if expected_coverage["is_full_day"]
        else {"completed_partial", "completed_partial_no_regular_session_rows"}
    )
    if result.get("status") not in expected_statuses:
        errors.append("research status does not match the source coverage")
    source = result.get("source")
    if not isinstance(source, dict) or source.get("manifest_sha256") != _sha256(source_path):
        errors.append("research source manifest hash does not match")
    elif source.get("gz_sha256") != source_manifest.get("gz_sha256"):
        errors.append("research fetched-gzip hash does not match")

    pipeline = result.get("pipeline")
    if not isinstance(pipeline, dict):
        errors.append("research pipeline provenance is invalid")
    elif (
        pipeline.get("source_script") != "python_quant/scripts/run_research.py"
        or pipeline.get("source_sha256") != _pipeline_fingerprint()
        or pipeline.get("experiments") != ["E1-E4", "E5", "E6"]
        or pipeline.get("regular_session_only") is not True
        or pipeline.get("horizons") != list(HORIZONS)
        or pipeline.get("fill_horizons_events") != list(FILL_HORIZONS)
        or pipeline.get("adverse_horizons_events") != list(ADVERSE_HORIZONS)
        or pipeline.get("n_boot") != N_BOOT
        or pipeline.get("pooled_cross_day_inference") is not False
    ):
        errors.append("research pipeline provenance does not match the E1–E6 batch")

    result_symbols = result.get("symbols") if isinstance(result.get("symbols"), dict) else {}
    source_slices = source.get("slices") if isinstance(source, dict) and isinstance(source.get("slices"), dict) else {}
    for symbol in symbols:
        output = result_symbols.get(symbol)
        if not isinstance(output, dict):
            errors.append(f"research output is missing {symbol}")
            continue
        filename = output.get("file")
        if filename != f"real_tape_{day}_{symbol}.json":
            errors.append(f"research output filename is invalid for {symbol}")
            continue
        output_path = day_dir / filename
        if not output_path.is_file() or not _is_sha256(output.get("sha256")) or _sha256(output_path) != output["sha256"]:
            errors.append(f"research output hash mismatch for {symbol}")
        if not isinstance(source_slices.get(symbol), dict) or source_slices[symbol].get("sha256") != source_manifest["symbols"][symbol]["sha256"]:
            errors.append(f"research source slice hash mismatch for {symbol}")

    markdown = result.get("markdown")
    if not isinstance(markdown, dict) or markdown.get("file") != markdown_path.name:
        errors.append("research markdown entry is invalid")
    elif not markdown_path.is_file() or not _is_sha256(markdown.get("sha256")) or _sha256(markdown_path) != markdown["sha256"]:
        errors.append("research markdown hash mismatch")
    return (result if not errors else None), errors


def _new_batch_manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "nexus-lob-itch-research-batch",
        "days": {},
    }


def _load_batch_manifest(path: Path) -> dict[str, Any]:
    prior = _read_json(path)
    if (
        prior is None
        or prior.get("schema_version") != SCHEMA_VERSION
        or prior.get("kind") != "nexus-lob-itch-research-batch"
        or not isinstance(prior.get("days"), dict)
    ):
        return _new_batch_manifest()
    return prior


def _summary_data(batch: dict[str, Any], results_dir: Path) -> dict[str, Any]:
    """Collect descriptive per-day rows without combining samples or estimates."""
    status_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []
    tracked = batch.get("days", {})
    for day in sorted(tracked):
        record = tracked[day]
        if not isinstance(record, dict):
            continue
        coverage = record.get("coverage") if isinstance(record.get("coverage"), dict) else {}
        status_rows.append({
            "day": day,
            "batch_status": record.get("status"),
            "analysis_status": record.get("analysis_status"),
            "coverage": coverage.get("kind"),
            "is_full_day": coverage.get("is_full_day"),
            "symbols": record.get("symbols", []),
            "error": record.get("error"),
        })
        result_path = results_dir / day / RESEARCH_MANIFEST_NAME
        result = _read_json(result_path)
        if (
            result is None
            or result.get("status") != record.get("analysis_status")
            or not isinstance(result.get("symbols"), dict)
        ):
            continue
        for symbol, output in result["symbols"].items():
            if not isinstance(output, dict):
                continue
            filename = output.get("file")
            if filename != f"real_tape_{day}_{symbol}.json":
                continue
            payload_path = results_dir / day / filename
            expected_hash = output.get("sha256")
            if (
                not payload_path.is_file()
                or not _is_sha256(expected_hash)
                or _sha256(payload_path) != expected_hash
            ):
                continue
            payload = _read_json(payload_path)
            if payload is None:
                continue
            tape = payload.get("tape", {})
            status_rows[-1].setdefault("regular_events", {})[symbol] = tape.get("regular_events")
            for row in payload.get("ic", {}).get("rows", []):
                if not isinstance(row, dict) or row.get("feature") != "lob_imbalance":
                    continue
                signal_rows.append({
                    "day": day,
                    "symbol": symbol,
                    "coverage": coverage.get("kind"),
                    "horizon_events": row.get("horizon_h"),
                    "n_test": row.get("n_test"),
                    "rank_ic_test": row.get("ic_test"),
                    "ci95_lo": row.get("ci95_lo"),
                    "ci95_hi": row.get("ci95_hi"),
                })
            fill = payload.get("fill", {})
            km = fill.get("km", {}) if isinstance(fill, dict) else {}
            tau = km.get("tau_events", []) if isinstance(km, dict) else []
            probs = km.get("p_fill", []) if isinstance(km, dict) else []
            fill_50 = None
            if 50 in tau:
                fill_50 = probs[tau.index(50)]
            adverse = payload.get("adverse", {})
            h5 = adverse.get("horizons", {}).get("5") or adverse.get("horizons", {}).get(5, {})
            overall = h5.get("overall", {}) if isinstance(h5, dict) else {}
            queue_rows.append({
                "day": day,
                "symbol": symbol,
                "coverage": coverage.get("kind"),
                "regular_orders": fill.get("n_orders_regular") if isinstance(fill, dict) else None,
                "passive_fills": adverse.get("n_fills") if isinstance(adverse, dict) else None,
                "km_p_fill_h50": fill_50,
                "adverse_h5_mean_drift": overall.get("mean_drift"),
                "adverse_h5_n": overall.get("n"),
            })
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "nexus-lob-itch-multi-day-summary",
        "updated_at_unix": int(time.time()),
        "scope": {
            "description": "Descriptive index of separate per-day E1-E6 studies; no event rows or estimates are pooled.",
            "pooled_cross_day_inference": False,
            "partial_prefix_policy": "Partial GZIP prefixes are labelled and are not full-day evidence.",
        },
        "day_status": status_rows,
        "lob_imbalance_test_ic": signal_rows,
        "queue_and_adverse_availability": queue_rows,
    }


def _number(value: Any, digits: int = 4) -> str:
    if value is None or not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _render_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Multi-day NASDAQ ITCH research batch summary",
        "",
        (
            "> This is a descriptive index of separate, time-ordered E1–E6 day studies. It does **not** pool event rows, "
            "fit a cross-day model, or make a cross-day performance claim. Partial GZIP prefixes are explicitly not full-day evidence."
        ),
        "",
        "## Batch status and coverage",
        "",
        "| day | batch status | analysis status | coverage | symbols | regular-session rows |",
        "|---|---|---|---|---|---|",
    ]
    for row in summary["day_status"]:
        regular = row.get("regular_events", {})
        if isinstance(regular, dict):
            regular_text = ", ".join(f"{s}: {n:,}" for s, n in regular.items() if isinstance(n, int)) or "—"
        else:
            regular_text = "—"
        symbols = ", ".join(row.get("symbols", [])) if isinstance(row.get("symbols"), list) else "—"
        error = row.get("error")
        status = str(row.get("batch_status") or "—")
        if error:
            status += " (see JSON error)"
        lines.append(
            f"| {row['day']} | {status} | {row.get('analysis_status') or '—'} | "
            f"{row.get('coverage') or '—'} | {symbols or '—'} | {regular_text} |"
        )

    lines += [
        "",
        "## E1 descriptive test-split index — L1 imbalance",
        "",
        "Each IC and confidence interval remains a per-day estimate. Rows marked `partial_gzip_prefix` are shown for traceability, not pooled inference.",
        "",
        "| day | symbol | coverage | h (events) | n test | rank IC test | 95% CI |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in summary["lob_imbalance_test_ic"]:
        lines.append(
            f"| {row['day']} | {row['symbol']} | {row.get('coverage') or '—'} | {row.get('horizon_events') or '—'} | "
            f"{row.get('n_test') or '—'} | {_number(row.get('rank_ic_test'))} | "
            f"[{_number(row.get('ci95_lo'))}, {_number(row.get('ci95_hi'))}] |"
        )

    lines += [
        "",
        "## E5/E6 availability index",
        "",
        "These fields expose what each separate tape can support; they are not aggregated estimates.",
        "",
        "| day | symbol | coverage | regular orders | passive fills | KM P(fill by 50 events) | h=5 adverse drift | n |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["queue_and_adverse_availability"]:
        lines.append(
            f"| {row['day']} | {row['symbol']} | {row.get('coverage') or '—'} | "
            f"{row.get('regular_orders') if row.get('regular_orders') is not None else '—'} | "
            f"{row.get('passive_fills') if row.get('passive_fills') is not None else '—'} | "
            f"{_number(row.get('km_p_fill_h50'))} | {_number(row.get('adverse_h5_mean_drift'), 2)} | "
            f"{row.get('adverse_h5_n') if row.get('adverse_h5_n') is not None else '—'} |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_progress_and_summary(batch: dict[str, Any], results_dir: Path) -> None:
    batch["updated_at_unix"] = int(time.time())
    _atomic_write_json(results_dir / BATCH_MANIFEST_NAME, batch)
    summary = _summary_data(batch, results_dir)
    _atomic_write_json(results_dir / SUMMARY_JSON_NAME, summary)
    _atomic_write_text(results_dir / SUMMARY_MD_NAME, _render_summary(summary))


def _record(
    batch: dict[str, Any],
    day: str,
    *,
    status: str,
    symbols: list[str],
    coverage: dict[str, Any] | None = None,
    analysis_status: str | None = None,
    error: dict[str, str] | None = None,
) -> None:
    days = batch["days"]
    prior = days.get(day)
    item = dict(prior) if isinstance(prior, dict) else {}
    item.update({
        "day": day,
        "status": status,
        "symbols": list(symbols),
        "updated_at_unix": int(time.time()),
    })
    if coverage is not None:
        item["coverage"] = coverage
    if analysis_status is not None:
        item["analysis_status"] = analysis_status
    if error is None:
        item.pop("error", None)
    else:
        item["error"] = error
    days[day] = item


def run_batch(
    *,
    days: list[str],
    symbols: list[str],
    out_dir: Path,
    results_dir: Path,
    max_gz_bytes: int | None = None,
    download_workers: int = 1,
    skip_existing: bool = False,
    base: str = fetch_itch.DEFAULT_BASE,
    fetcher: Callable[..., dict[str, Any]] | None = None,
    research_runner: Callable[..., dict[str, Any]] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Process dates sequentially and return an explicit nonzero-compatible status.

    A valid local source is never re-downloaded.  ``--skip-existing`` additionally
    reuses a completed analysis only after both source and result hashes validate;
    otherwise missing/corrupt research is regenerated from the existing slices.
    """
    if not days:
        raise ValueError("at least one day is required")
    if not symbols:
        raise ValueError("at least one symbol is required")
    if download_workers not in (1, 2, 3, 4):
        raise ValueError("download_workers must be between 1 and 4")
    results_dir.mkdir(parents=True, exist_ok=True)
    batch = _load_batch_manifest(results_dir / BATCH_MANIFEST_NAME)
    batch["latest_request"] = {
        "days": list(days),
        "symbols": list(symbols),
        "max_gz_bytes": max_gz_bytes,
        "download_workers": download_workers,
        "base": base,
        "skip_existing": skip_existing,
    }
    _write_progress_and_summary(batch, results_dir)
    failures = 0
    runner = research_runner or run_day_research

    for day in days:
        try:
            source_manifest, source_errors = validate_fetch_manifest(
                day, symbols, out_dir, max_gz_bytes=max_gz_bytes, base=base
            )
            if source_manifest is None:
                if source_errors:
                    log(f"{day}: local slices not reusable ({'; '.join(source_errors)}); fetching into staging")
                _record(
                    batch,
                    day,
                    status="fetching",
                    symbols=symbols,
                    analysis_status="fetching",
                )
                _write_progress_and_summary(batch, results_dir)
                source_manifest = fetch_and_promote(
                    day, symbols, out_dir, max_gz_bytes=max_gz_bytes, base=base,
                    download_workers=download_workers, fetcher=fetcher, log=log
                )
                source_state = "slices_downloaded"
            else:
                source_state = "slices_reused"
            coverage = _coverage(source_manifest)

            existing, result_errors = validate_research_manifest(day, symbols, out_dir, results_dir, source_manifest)
            if skip_existing and existing is not None:
                existing_status = str(existing.get("status"))
                _record(
                    batch,
                    day,
                    status=(
                        "skipped_existing_partial"
                        if existing_status.startswith("completed_partial")
                        else "skipped_existing"
                    ),
                    symbols=symbols,
                    coverage=coverage,
                    analysis_status=existing_status,
                )
                _write_progress_and_summary(batch, results_dir)
                log(f"{day}: valid source and E1–E6 result reused")
                continue
            if result_errors:
                log(f"{day}: regenerating research ({'; '.join(result_errors)})")

            _record(
                batch,
                day,
                status="researching",
                symbols=symbols,
                coverage=coverage,
                analysis_status="researching",
            )
            _write_progress_and_summary(batch, results_dir)
            research = runner(
                day, symbols, out_dir, results_dir, source_manifest, log=log
            )
            _record(
                batch,
                day,
                status=str(research["status"]),
                symbols=symbols,
                coverage=coverage,
                analysis_status=str(research["status"]),
            )
            _write_progress_and_summary(batch, results_dir)
            log(f"{day}: {research['status']} ({source_state})")
        except KeyboardInterrupt:
            _record(
                batch,
                day,
                status="interrupted",
                symbols=symbols,
                analysis_status="interrupted",
            )
            _write_progress_and_summary(batch, results_dir)
            raise
        except Exception as exc:  # noqa: BLE001 - independent days should preserve each failure and continue.
            failures += 1
            error = {"type": type(exc).__name__, "message": str(exc)}
            _record(
                batch,
                day,
                status="failed",
                symbols=symbols,
                analysis_status="failed",
                error=error,
            )
            _write_progress_and_summary(batch, results_dir)
            log(f"{day}: FAILED — {error['type']}: {error['message']}")

    batch["finished_at_unix"] = int(time.time())
    _write_progress_and_summary(batch, results_dir)
    return {"exit_code": 1 if failures else 0, "failures": failures, "manifest": batch}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", required=True, help="comma-separated PUBLIC_SAMPLE_DAYS dates, or 'all'")
    parser.add_argument("--symbols", default="AAPL,QQQ", help="comma-separated tickers (default: AAPL,QQQ)")
    parser.add_argument("--max-gz-bytes", type=_positive_int, default=None, help="bounded GZIP prefix; always labelled partial")
    parser.add_argument("--download-workers", type=int, choices=range(1, 5), default=1,
                        help="full-tape range workers (2–4 enable resumable compressed-byte caching)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/itch"), help="source slice root (default: data/itch)")
    parser.add_argument(
        "--results-dir", type=Path, default=Path("docs/results/multi_day"),
        help="batch result root (default: docs/results/multi_day)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="reuse a day only when its source and E1–E6 manifests/hashes validate",
    )
    parser.add_argument("--base", default=fetch_itch.DEFAULT_BASE, help="public ITCH base URL (testing/mirror override)")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.days = parse_days(args.days)
        args.symbols = parse_symbols(args.symbols)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outcome = run_batch(
        days=args.days,
        symbols=args.symbols,
        out_dir=args.out_dir,
        results_dir=args.results_dir,
        max_gz_bytes=args.max_gz_bytes,
        download_workers=args.download_workers,
        skip_existing=args.skip_existing,
        base=args.base,
    )
    if outcome["failures"]:
        print(f"batch finished with {outcome['failures']} failed day(s); inspect {args.results_dir / BATCH_MANIFEST_NAME}")
    else:
        print(f"batch finished; inspect {args.results_dir / SUMMARY_MD_NAME}")
    return int(outcome["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())

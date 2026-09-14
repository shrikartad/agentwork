"""Offline coverage for the resumable multi-day ITCH batch runner."""
from __future__ import annotations

import gzip
import io
import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "python_quant" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import batch_research_itch as batch
from nexus_quant.book_state import Side
from nexus_quant.itch_parser import EventType, NormalizedEvent, encode_event

DAYS = ("12302019", "01302020")


def _framed(body: bytes) -> bytes:
    return len(body).to_bytes(2, "big") + body


def _system_event(code: bytes, ts: int) -> bytes:
    return _framed(b"S" + (0).to_bytes(2, "big") + (0).to_bytes(2, "big") + ts.to_bytes(6, "big") + code)


def _directory(locate: int, stock: str, ts: int) -> bytes:
    body = (
        b"R" + locate.to_bytes(2, "big") + (0).to_bytes(2, "big") + ts.to_bytes(6, "big")
        + stock.ljust(8).encode() + b"Q" + b"N" + (100).to_bytes(4, "big") + b"N" + b"Q" + b"N"
        + b"  " + b"N" + b"N" + b" " + b"N" + (0).to_bytes(4, "big") + b"N"
    )
    assert len(body) == 39
    return _framed(body)


def _add(locate: int, order_id: int, side: Side, price: int, size: int, ts: int) -> bytes:
    raw = bytearray(
        encode_event(
            NormalizedEvent(EventType.ADD, ts, order_id, side, price, size),
            framed=False,
        )
    )
    raw[1:3] = locate.to_bytes(2, "big")
    return _framed(bytes(raw))


def _tape(symbol: str = "AAPL", n_events: int = 12) -> bytes:
    """A deterministic regular-session slice sufficient to exercise E1–E6 wiring."""
    ts = 34_200_000_000_000
    messages = [_system_event(b"O", ts - 2), _directory(7, symbol, ts - 1), _system_event(b"Q", ts)]
    for i in range(1, n_events + 1):
        messages.append(_add(7, i, Side.Bid if i % 2 else Side.Ask, 10_000 if i % 2 else 10_010, 100, ts + i))
    return b"".join(messages)


def _pre_open_tape(symbol: str = "AAPL") -> bytes:
    """A valid slice whose prefix ends before the regular session opens."""
    ts = 34_199_000_000_000
    return b"".join([_system_event(b"O", ts - 2), _directory(7, symbol, ts - 1), _system_event(b"Q", ts)])


def _install_urlopen(
    monkeypatch: pytest.MonkeyPatch,
    payloads: dict[str, bytes],
    *,
    fail: Callable[[str], bool] | None = None,
) -> list[str]:
    calls: list[str] = []

    def fake_urlopen(request, timeout: int):
        url = request.full_url
        calls.append(url)
        if fail is not None and fail(url):
            raise OSError("fixture download interrupted")
        return io.BytesIO(payloads[url])

    monkeypatch.setattr(batch.fetch_itch.urllib.request, "urlopen", fake_urlopen)
    return calls


def _payloads() -> dict[str, bytes]:
    compressed = gzip.compress(_tape())
    return {batch.fetch_itch.tape_url(day): compressed for day in DAYS}


def test_parse_args_validates_catalogued_days_symbols_and_limits() -> None:
    args = batch.parse_args(["--days", "12302019,01302020", "--symbols", "aapl, qqq, AAPL", "--max-gz-bytes", "64"])
    assert args.days == list(DAYS)
    assert args.symbols == ["AAPL", "QQQ"]
    assert args.max_gz_bytes == 64
    assert batch.parse_days("all") == list(batch.fetch_itch.PUBLIC_SAMPLE_DAYS)
    with pytest.raises(SystemExit):
        batch.parse_args(["--days", "02302019"])
    with pytest.raises(SystemExit):
        batch.parse_args(["--days", "12312021"])
    with pytest.raises(SystemExit):
        batch.parse_args(["--days", "12302019", "--max-gz-bytes", "0"])


def test_two_days_stream_slice_and_run_existing_e1_to_e6_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_urlopen(monkeypatch, _payloads())
    out_dir, results_dir = tmp_path / "itch", tmp_path / "results"

    outcome = batch.run_batch(
        days=list(DAYS),
        symbols=["AAPL"],
        out_dir=out_dir,
        results_dir=results_dir,
        log=lambda _: None,
    )

    assert outcome["exit_code"] == 0
    assert len(calls) == 2
    for day in DAYS:
        source, errors = batch.validate_fetch_manifest(day, ["AAPL"], out_dir, max_gz_bytes=None)
        assert errors == [] and source is not None
        assert source["gzip_stream_complete"] is True
        result = json.loads((results_dir / day / batch.RESEARCH_MANIFEST_NAME).read_text())
        assert result["status"] == "completed"
        assert result["pipeline"]["experiments"] == ["E1-E4", "E5", "E6"]
        assert (results_dir / day / result["symbols"]["AAPL"]["file"]).is_file()
    summary = json.loads((results_dir / batch.SUMMARY_JSON_NAME).read_text())
    assert {row["day"] for row in summary["day_status"]} == set(DAYS)
    summary_md = (results_dir / batch.SUMMARY_MD_NAME).read_text()
    assert "does **not** pool event rows" in summary_md
    assert "cross-day performance claim" in summary_md


def test_small_deterministic_slice_runs_e1_to_e6_not_just_the_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = gzip.compress(_tape(n_events=610))
    _install_urlopen(monkeypatch, {batch.fetch_itch.tape_url(DAYS[0]): payload})
    out_dir, results_dir = tmp_path / "itch", tmp_path / "results"

    outcome = batch.run_batch(
        days=[DAYS[0]],
        symbols=["AAPL"],
        out_dir=out_dir,
        results_dir=results_dir,
        log=lambda _: None,
    )

    assert outcome["exit_code"] == 0
    output = json.loads((results_dir / DAYS[0] / f"real_tape_{DAYS[0]}_AAPL.json").read_text())
    assert output["tape"]["regular_events"] == 610
    assert {row["horizon_h"] for row in output["ic"]["rows"]} == set(batch.HORIZONS)
    assert {row["feature"] for row in output["ic"]["rows"]} == {
        "lob_imbalance",
        "microprice_off",
        "deep_imbalance",
        "spread_bps",
        "ofi_l2",
        "ofi_order",
        "ofi_order_w20",
    }
    assert output["fill"]["n_orders_regular"] == 610
    assert output["adverse"]["n_fills"] == 0


def test_queue_and_adverse_results_include_actual_offline_fills(tmp_path, monkeypatch):
    raw = _tape(n_events=610)
    # E5/E6 consume resolved orders; include both fills and competing cancels.
    for order_id in range(1, 301):
        event = NormalizedEvent(
            EventType.EXECUTE if order_id % 3 == 0 else EventType.DELETE,
            34_200_000_001_000 + order_id,
            order_id=order_id, side=Side.NONE, price_ticks=0, size=100,
        )
        body = bytearray(encode_event(event, framed=False))
        body[1:3] = (7).to_bytes(2, "big")
        raw += _framed(bytes(body))
    _install_urlopen(monkeypatch, {batch.fetch_itch.tape_url(DAYS[0]): gzip.compress(raw)})
    out_dir, results_dir = tmp_path / "itch", tmp_path / "results"
    outcome = batch.run_batch(
        days=[DAYS[0]], symbols=["AAPL"], out_dir=out_dir, results_dir=results_dir, log=lambda _: None,
    )
    assert outcome["exit_code"] == 0
    result = json.loads((results_dir / DAYS[0] / f"real_tape_{DAYS[0]}_AAPL.json").read_text())
    assert result["adverse"]["n_fills"] == 100
    assert result["adverse"]["horizons"]["5"]["overall"]["n"] > 0
    assert "logistic" in result["fill"]
    assert max(result["fill"]["km"]["p_fill"]) > 0
    summary = json.loads((results_dir / batch.SUMMARY_JSON_NAME).read_text())
    assert summary["queue_and_adverse_availability"][0]["passive_fills"] == 100


def test_changed_pipeline_regenerates_research_without_refetching(tmp_path, monkeypatch):
    calls = _install_urlopen(monkeypatch, _payloads())
    kwargs = {
        "days": [DAYS[0]], "symbols": ["AAPL"], "out_dir": tmp_path / "itch",
        "results_dir": tmp_path / "results", "skip_existing": True, "log": lambda _: None,
    }
    assert batch.run_batch(**kwargs)["exit_code"] == 0
    assert batch.run_batch(**kwargs)["manifest"]["days"][DAYS[0]]["status"] == "skipped_existing"
    monkeypatch.setattr(batch, "_pipeline_fingerprint", lambda: "0" * 64)
    result = batch.run_batch(**kwargs)
    assert result["exit_code"] == 0
    assert result["manifest"]["days"][DAYS[0]]["status"] == "completed"
    assert len(calls) == 1
    manifest_path = kwargs["results_dir"] / DAYS[0] / batch.RESEARCH_MANIFEST_NAME
    assert json.loads(manifest_path.read_text())["pipeline"]["source_sha256"] == "0" * 64


@pytest.mark.parametrize("change", ["day", "url", "size", "hash", "missing_slice", "missing_symbol", "filename"])
def test_source_manifest_rejects_missing_or_mismatched_slices(tmp_path, monkeypatch, change):
    _install_urlopen(monkeypatch, _payloads())
    out_dir = tmp_path / "itch"
    batch.fetch_itch.fetch(day=DAYS[0], symbols={"AAPL"}, out_root=out_dir, log=lambda _: None)
    path = out_dir / DAYS[0] / "manifest.json"
    manifest = json.loads(path.read_text())
    if change == "day":
        manifest["day"] = DAYS[1]
    elif change == "url":
        manifest["source_url"] = batch.fetch_itch.tape_url(DAYS[1])
    elif change == "size":
        manifest["symbols"]["AAPL"]["bytes"] += 1
    elif change == "hash":
        manifest["symbols"]["AAPL"]["sha256"] = "0" * 64
    elif change == "missing_slice":
        (out_dir / DAYS[0] / "AAPL.itch").unlink()
    elif change == "missing_symbol":
        del manifest["symbols"]["AAPL"]
    else:
        manifest["symbols"]["AAPL"]["file"] = "../AAPL.itch"
    path.write_text(json.dumps(manifest))
    valid, errors = batch.validate_fetch_manifest(DAYS[0], ["AAPL"], out_dir, max_gz_bytes=None)
    assert valid is None
    assert errors


def test_partial_prefix_is_labelled_partial_and_request_limit_is_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = gzip.compress(_tape())
    # Stop inside the gzip trailer: all early ITCH messages inflate, but the fetch
    # cannot prove a full source stream and must remain a partial-prefix result.
    cap = len(payload) - 1
    calls = _install_urlopen(monkeypatch, {batch.fetch_itch.tape_url(DAYS[0]): payload})
    out_dir, results_dir = tmp_path / "itch", tmp_path / "results"

    outcome = batch.run_batch(
        days=[DAYS[0]],
        symbols=["AAPL"],
        out_dir=out_dir,
        results_dir=results_dir,
        max_gz_bytes=cap,
        log=lambda _: None,
    )

    assert outcome["exit_code"] == 0
    source, errors = batch.validate_fetch_manifest(DAYS[0], ["AAPL"], out_dir, max_gz_bytes=cap)
    assert errors == [] and source is not None
    assert source["gz_bytes_fetched"] == cap  # locally enforced even if Range is ignored
    assert source["gzip_stream_complete"] is False
    result = json.loads((results_dir / DAYS[0] / batch.RESEARCH_MANIFEST_NAME).read_text())
    assert result["status"] == "completed_partial"
    assert result["coverage"]["is_full_day"] is False
    state = json.loads((results_dir / batch.BATCH_MANIFEST_NAME).read_text())
    assert state["days"][DAYS[0]]["status"] == "completed_partial"
    assert "Partial GZIP prefix — not a full-day result" in (results_dir / DAYS[0] / f"real_tape_{DAYS[0]}.md").read_text()

    research_path = results_dir / DAYS[0] / batch.RESEARCH_MANIFEST_NAME
    research = json.loads(research_path.read_text())
    research["status"] = "completed"  # A partial source must never be reused as a full-day claim.
    research_path.write_text(json.dumps(research))
    reused_research, research_errors = batch.validate_research_manifest(
        DAYS[0], ["AAPL"], out_dir, results_dir, source
    )
    assert reused_research is None
    assert any("status does not match" in error for error in research_errors)

    resumed = batch.run_batch(
        days=[DAYS[0]],
        symbols=["AAPL"],
        out_dir=out_dir,
        results_dir=results_dir,
        max_gz_bytes=cap,
        skip_existing=True,
        log=lambda _: None,
    )
    assert resumed["exit_code"] == 0
    assert len(calls) == 1  # Existing, valid source slices were not downloaded again.

    manifest_path = out_dir / DAYS[0] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["max_gz_bytes_requested"] = cap + 1
    manifest_path.write_text(json.dumps(manifest))
    reused, errors = batch.validate_fetch_manifest(DAYS[0], ["AAPL"], out_dir, max_gz_bytes=cap)
    assert reused is None
    assert any("max_gz_bytes_requested" in error for error in errors)


def test_partial_prefix_with_no_regular_session_rows_is_not_a_full_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = gzip.compress(_pre_open_tape())
    _install_urlopen(monkeypatch, {batch.fetch_itch.tape_url(DAYS[0]): payload})
    out_dir, results_dir = tmp_path / "itch", tmp_path / "results"

    outcome = batch.run_batch(
        days=[DAYS[0]],
        symbols=["AAPL"],
        out_dir=out_dir,
        results_dir=results_dir,
        max_gz_bytes=len(payload) - 1,
        log=lambda _: None,
    )

    assert outcome["exit_code"] == 0
    research = json.loads((results_dir / DAYS[0] / batch.RESEARCH_MANIFEST_NAME).read_text())
    assert research["status"] == "completed_partial_no_regular_session_rows"
    assert research["coverage"]["kind"] == "partial_gzip_prefix"
    output = json.loads((results_dir / DAYS[0] / research["symbols"]["AAPL"]["file"]).read_text())
    assert output["tape"]["regular_events"] == 0


def test_failed_day_resumes_without_redownloading_completed_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = _payloads()
    failed_once = {DAYS[0]}

    def fail_first(url: str) -> bool:
        day = next(day for day in DAYS if day in url)
        if day in failed_once:
            failed_once.remove(day)
            return True
        return False

    calls = _install_urlopen(monkeypatch, payloads, fail=fail_first)
    out_dir, results_dir = tmp_path / "itch", tmp_path / "results"
    first = batch.run_batch(days=list(DAYS), symbols=["AAPL"], out_dir=out_dir, results_dir=results_dir, log=lambda _: None)
    assert first["exit_code"] == 1
    state = json.loads((results_dir / batch.BATCH_MANIFEST_NAME).read_text())
    assert state["days"][DAYS[0]]["status"] == "failed"
    assert state["days"][DAYS[1]]["status"] == "completed"
    second_day_calls = sum(DAYS[1] in url for url in calls)

    second = batch.run_batch(
        days=list(DAYS), symbols=["AAPL"], out_dir=out_dir, results_dir=results_dir, skip_existing=True, log=lambda _: None
    )
    assert second["exit_code"] == 0
    assert sum(DAYS[1] in url for url in calls) == second_day_calls
    state = json.loads((results_dir / batch.BATCH_MANIFEST_NAME).read_text())
    assert state["days"][DAYS[0]]["status"] == "completed"
    assert state["days"][DAYS[1]]["status"] == "skipped_existing"


def test_midstream_fetch_failure_does_not_promote_partial_slices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingStream(io.BytesIO):
        def __init__(self, payload: bytes) -> None:
            super().__init__(payload)
            self._read_once = False

        def read(self, size: int = -1) -> bytes:
            if self._read_once:
                raise OSError("fixture stream disconnected")
            self._read_once = True
            return super().read(size)

    payload = gzip.compress(_tape())

    def fake_urlopen(request, timeout: int) -> FailingStream:
        assert request.full_url == batch.fetch_itch.tape_url(DAYS[0])
        return FailingStream(payload)

    monkeypatch.setattr(batch.fetch_itch.urllib.request, "urlopen", fake_urlopen)
    out_dir, results_dir = tmp_path / "itch", tmp_path / "results"
    outcome = batch.run_batch(
        days=[DAYS[0]],
        symbols=["AAPL"],
        out_dir=out_dir,
        results_dir=results_dir,
        log=lambda _: None,
    )

    assert outcome["exit_code"] == 1
    assert not (out_dir / DAYS[0]).exists()
    assert not list(out_dir.glob(f".{DAYS[0]}.fetch-*"))
    state = json.loads((results_dir / batch.BATCH_MANIFEST_NAME).read_text())
    assert state["days"][DAYS[0]]["status"] == "failed"
    assert state["days"][DAYS[0]]["error"]["type"] == "OSError"


def test_interrupted_research_leaves_resumable_source_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_urlopen(monkeypatch, {batch.fetch_itch.tape_url(DAYS[0]): gzip.compress(_tape())})
    out_dir, results_dir = tmp_path / "itch", tmp_path / "results"

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        batch.run_batch(
            days=[DAYS[0]],
            symbols=["AAPL"],
            out_dir=out_dir,
            results_dir=results_dir,
            research_runner=interrupt,
            log=lambda _: None,
        )
    state = json.loads((results_dir / batch.BATCH_MANIFEST_NAME).read_text())
    assert state["days"][DAYS[0]]["status"] == "interrupted"
    assert (out_dir / DAYS[0] / "manifest.json").is_file()

    resumed = batch.run_batch(
        days=[DAYS[0]],
        symbols=["AAPL"],
        out_dir=out_dir,
        results_dir=results_dir,
        skip_existing=True,
        log=lambda _: None,
    )
    assert resumed["exit_code"] == 0
    assert len(calls) == 1  # source was kept; only missing E1–E6 output was regenerated

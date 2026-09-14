#!/usr/bin/env python3
"""Fetch a public NASDAQ TotalView-ITCH 5.0 sample day and slice it per symbol.

Source / provenance
-------------------
NASDAQ publishes full-day ITCH 5.0 sample files for evaluation at

    https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/<MMDDYYYY>.NASDAQ_ITCH50.gz

(e.g. ``12302019.NASDAQ_ITCH50.gz``, ~3.5 GB gzipped, ~10 GB raw, ~300 M
messages). The files are 2-byte big-endian length-prefixed ITCH 5.0 messages.

License note: the sample files are provided by Nasdaq for evaluation of the
TotalView-ITCH product. They are **not** redistributed by this repository —
``data/`` and ``*.itch`` are gitignored and every artifact this script writes
carries a ``manifest.json`` with the exact source URL, byte range, and SHA-256
so a run is reproducible from the public source. Do not commit the output.

What this script does
---------------------
It streams the gzip over HTTP (optionally a bounded ``Range`` prefix for smoke
tests), inflates it on the fly, frames every message, and writes **one framed
``.itch`` file per requested symbol** containing:

* every ``S`` (System Event) message — session boundaries for all symbols;
* the symbol's ``R`` (Stock Directory) message — locate → symbol proof;
* every order/print message for that locate code: ``A F E C X D U P``.

Everything else (other symbols, NOII, cross, admin) is dropped, which is what
turns a 10 GB day into a few tens of MB per liquid symbol that
``iter_itch_events`` / ``ReplayEngine`` consume in seconds.

Run (from repo root)::

    python python_quant/scripts/fetch_itch.py --day 12302019 --symbols QQQ,AAPL
    # smoke: only the first 64 MB of the gzip (pre-market + directory)
    python python_quant/scripts/fetch_itch.py --day 12302019 --symbols QQQ --max-gz-bytes 67108864

Output: ``data/itch/<day>/<SYMBOL>.itch`` + ``data/itch/<day>/manifest.json``.
No third-party dependency — stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
import zlib
from contextlib import closing
from pathlib import Path

DEFAULT_BASE = "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/"
DEFAULT_DAY = "12302019"  # smallest recent full day on the public server
DEFAULT_SYMBOLS = ("QQQ", "AAPL")

# Full-session gzip files verified in Nasdaq's directory on 2026-09-14.
# 2025-11-28 is a half-day and is excluded from the fixed 09:30–16:00 study.
PUBLIC_SAMPLE_FILES: dict[str, str] = {
    "01302019": "01302019.NASDAQ_ITCH50.gz",
    "03272019": "03272019.NASDAQ_ITCH50.gz",
    "07302019": "07302019.NASDAQ_ITCH50.gz",
    "08302019": "08302019.NASDAQ_ITCH50.gz",
    "10182019": "S101819-v50.txt.gz",
    "10302019": "10302019.NASDAQ_ITCH50.gz",
    "12302019": "12302019.NASDAQ_ITCH50.gz",
    "01302020": "01302020.NASDAQ_ITCH50.gz",
    "07132021": "S071321-v50.txt.gz",
    "08132021": "S081321-v50.txt.gz",
    "12082025": "S120825-v50.txt.gz",
    "12092025": "S120925-v50.txt.gz",
    "12102025": "S121025-v50.txt.gz",
    "12112025": "S121125-v50.txt.gz",
    "12122025": "S121225-v50.txt.gz",
}
PUBLIC_SAMPLE_DAYS: tuple[str, ...] = tuple(PUBLIC_SAMPLE_FILES)

# Only checksum stubs remain online; explicit dates still permit local reuse.
UNAVAILABLE_SAMPLE_DAYS: tuple[str, ...] = (
    "01302018",
    "03292018",
    "05302018",
    "05302019",
    "07302018",
    "08302018",
    "10302018",
    "12282018",
)

# Message types carried over into every per-symbol slice.
_ORDER_TYPES = frozenset(b"AFECXDUP")
_SESSION_TYPE = ord("S")
_DIRECTORY_TYPE = ord("R")


def tape_url(day: str, base: str = DEFAULT_BASE) -> str:
    filename = PUBLIC_SAMPLE_FILES.get(day, f"{day}.NASDAQ_ITCH50.gz")
    return f"{base}{filename}"


class _SymbolSlicer:
    """Frames a raw ITCH byte stream and fans messages out per locate code."""

    def __init__(self, out_dir: Path, symbols: set[str]) -> None:
        self.out_dir = out_dir
        self.wanted = {s.upper() for s in symbols}
        self.locate_to_symbol: dict[int, str] = {}
        self.files: dict[int, object] = {}
        self.counts: dict[str, dict[str, int]] = {s: {} for s in self.wanted}
        self.session_msgs: list[bytes] = []  # S messages seen so far (replayed into late-opened files)
        self.messages = 0
        self.raw_bytes = 0
        self.last_ts_ns = 0
        self._buf = bytearray()

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk
        self.raw_bytes += len(chunk)
        buf = self._buf
        n = len(buf)
        i = 0
        while i + 2 <= n:
            ln = (buf[i] << 8) | buf[i + 1]
            end = i + 2 + ln
            if end > n:
                break
            typ = buf[i + 2]
            locate = (buf[i + 3] << 8) | buf[i + 4]
            self.messages += 1
            if typ == _SESSION_TYPE:
                msg = bytes(buf[i:end])
                self.last_ts_ns = int.from_bytes(msg[7:13], "big")
                self.session_msgs.append(msg)
                for fh in self.files.values():
                    fh.write(msg)  # type: ignore[attr-defined]
            elif typ == _DIRECTORY_TYPE:
                stock = bytes(buf[i + 13 : i + 21]).decode("ascii", "replace").strip()
                if stock in self.wanted and locate not in self.files:
                    self.locate_to_symbol[locate] = stock
                    fh = open(self.out_dir / f"{stock}.itch", "wb")  # noqa: SIM115 — long-lived handle
                    fh.writelines(self.session_msgs)
                    fh.write(bytes(buf[i:end]))
                    self.files[locate] = fh
                    self.counts[stock]["R"] = 1
            elif typ in _ORDER_TYPES and locate in self.files:
                self.files[locate].write(bytes(buf[i:end]))  # type: ignore[attr-defined]
                c = self.counts[self.locate_to_symbol[locate]]
                key = chr(typ)
                c[key] = c.get(key, 0) + 1
            i = end
        del buf[:i]

    def close(self) -> None:
        for fh in self.files.values():
            fh.close()  # type: ignore[attr-defined]
        self.files.clear()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _stream_blocks(req, *, chunk: int, max_bytes: int | None):
    total = 0
    with urllib.request.urlopen(req, timeout=120) as response:
        while max_bytes is None or total < max_bytes:
            size = chunk if max_bytes is None else min(chunk, max_bytes - total)
            block = response.read(size)
            if not block:
                break
            total += len(block)
            yield block


def fetch(
    *,
    day: str,
    symbols: set[str],
    out_root: Path,
    base: str = DEFAULT_BASE,
    max_gz_bytes: int | None = None,
    chunk: int = 1 << 20,
    download_workers: int = 1,
    download_cache: Path | None = None,
    log=print,
) -> dict:
    """Stream ``day`` from ``base``, slice ``symbols`` into ``out_root/day/``."""
    if max_gz_bytes is not None and max_gz_bytes <= 0:
        raise ValueError("max_gz_bytes must be greater than zero")
    if download_workers not in (1, 2, 3, 4):
        raise ValueError("download_workers must be between 1 and 4")
    url = tape_url(day, base)
    out_dir = out_root / day
    out_dir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "nexus-lob-fetch-itch/1.0"})
    if max_gz_bytes is not None:
        req.add_header("Range", f"bytes=0-{int(max_gz_bytes) - 1}")

    ranged = download_workers > 1 and max_gz_bytes is None
    cache = download_cache if download_cache is not None else out_root / ".downloads" / day
    source = None
    if ranged:
        from itch_transport import iter_cached_ranges, probe_source

        source = probe_source(url, log=log)
        blocks = iter_cached_ranges(url, cache, source=source, workers=download_workers, log=log)
    else:
        blocks = _stream_blocks(req, chunk=chunk, max_bytes=max_gz_bytes)

    slicer = _SymbolSlicer(out_dir, symbols)
    inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
    gz_bytes = 0
    gz_hash = hashlib.sha256()
    gz_md5 = hashlib.md5(usedforsecurity=False)
    t0 = time.time()
    next_log = 0
    try:
        with closing(blocks):
            for block in blocks:
                gz_bytes += len(block)
                gz_hash.update(block)
                gz_md5.update(block)
                slicer.feed(inflater.decompress(block))
                if gz_bytes >= next_log:
                    hours = slicer.last_ts_ns / 3.6e12
                    log(
                        f"  {gz_bytes / 1e6:8.1f} MB gz · {slicer.messages / 1e6:7.2f} M msgs · "
                        f"tape clock {hours:5.2f} h · {time.time() - t0:6.1f} s"
                    )
                    next_log += 256 << 20
            # A Range-truncated gzip has no valid trailer — flush what inflated cleanly.
            try:
                slicer.feed(inflater.flush())
            except zlib.error:
                pass
    finally:
        slicer.close()

    outputs = {}
    for locate, stock in slicer.locate_to_symbol.items():
        p = out_dir / f"{stock}.itch"
        outputs[stock] = {
            "file": p.name,
            "locate": locate,
            "bytes": p.stat().st_size,
            "sha256": _sha256(p),
            "messages": slicer.counts[stock],
        }
    missing = sorted(slicer.wanted - set(outputs))
    manifest = {
        "source_url": url,
        "day": day,
        "gz_bytes_fetched": gz_bytes,
        "range_limited": max_gz_bytes is not None,
        "max_gz_bytes_requested": max_gz_bytes,
        "gzip_stream_complete": inflater.eof,
        "gz_sha256": gz_hash.hexdigest(),
        "gz_md5": gz_md5.hexdigest(),
        "download_workers": download_workers if ranged else 1,
        "http_source_identity": source.as_dict() if source is not None else None,
        "raw_bytes_inflated": slicer.raw_bytes,
        "messages_framed": slicer.messages,
        "last_tape_ts_ns": slicer.last_ts_ns,
        "session_events": [
            {"code": chr(message[13]), "ts_ns": int.from_bytes(message[7:13], "big")}
            for message in slicer.session_msgs
        ],
        "symbols": outputs,
        "symbols_not_found": missing,
        "fetched_at_unix": int(time.time()),
        "license": (
            "Nasdaq TotalView-ITCH sample data, provided by Nasdaq for evaluation; "
            "not redistributed by this repository (data/ is gitignored)."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"done: {gz_bytes / 1e6:.1f} MB gz → {slicer.messages / 1e6:.2f} M msgs in {time.time() - t0:.1f} s")
    for stock, info in outputs.items():
        log(f"  {stock:<6} {info['bytes'] / 1e6:7.2f} MB  {info['messages']}")
    if missing:
        log(f"  not found in directory: {missing}")
    if ranged and inflater.eof and not missing:
        from itch_transport import cleanup_cached_ranges

        cleanup_cached_ranges(cache, source)
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", default=DEFAULT_DAY, help="MMDDYYYY of the public sample (default 12302019)")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="comma-separated tickers")
    ap.add_argument("--out", type=Path, default=Path("data/itch"), help="output root (gitignored)")
    ap.add_argument("--base", default=DEFAULT_BASE, help="override the public base URL")
    ap.add_argument("--download-workers", type=int, choices=range(1, 5), default=1,
                    help="full-tape range workers (2–4 enable a resumable compressed-byte cache)")
    ap.add_argument(
        "--max-gz-bytes", type=int, default=None,
        help="only fetch this many gzip bytes (HTTP Range) — smoke tests / CI",
    )
    args = ap.parse_args(argv)
    symbols = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
    if not symbols:
        ap.error("--symbols must name at least one ticker")
    print(f"fetching {tape_url(args.day, args.base)} → {args.out / args.day}  symbols={sorted(symbols)}")
    fetch(day=args.day, symbols=symbols, out_root=args.out, base=args.base,
          max_gz_bytes=args.max_gz_bytes, download_workers=args.download_workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())

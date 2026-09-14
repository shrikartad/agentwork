"""Offline tests for the resumable HTTP Range ITCH transport."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import itch_transport


class _Response:
    def __init__(
        self,
        *,
        status: int,
        headers: dict[str, str],
        body: bytes = b"",
        url: str,
        read_delay: float = 0.0,
        on_close=None,
    ) -> None:
        self.status = status
        self.headers = headers
        self._body = body
        self._offset = 0
        self._url = url
        self._read_delay = read_delay
        self._on_close = on_close
        self._closed = False

    def geturl(self) -> str:
        return self._url

    def read(self, size: int = -1) -> bytes:
        if self._read_delay:
            time.sleep(self._read_delay)
        if size is None or size < 0:
            size = len(self._body) - self._offset
        block = self._body[self._offset : self._offset + size]
        self._offset += len(block)
        return block

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._on_close is not None:
                self._on_close()


class _RangeServer:
    """A thread-safe urlopen replacement with controllable HTTP behavior."""

    def __init__(
        self,
        data: bytes,
        *,
        url: str = "https://tape.example.test/day.gz",
        etag: str | None = '"version-1"',
        last_modified: str | None = "Mon, 14 Sep 2026 00:00:00 GMT",
    ) -> None:
        self.data = data
        self.url = url
        self.etag = etag
        self.last_modified = last_modified
        self.get_requests: list[dict[str, str | None]] = []
        self.head_requests = 0
        self.max_active = 0
        self._active = 0
        self._lock = threading.Lock()
        self.read_delay = 0.0
        self.status_override: int | None = None
        self.content_range_override: str | None = None
        self.body_transform = None
        self.response_etags: dict[int, str | None] = {}
        self.transient_failures_remaining = 0

    def __call__(self, request, *, timeout: float):
        del timeout
        method = request.get_method()
        if method == "HEAD":
            self.head_requests += 1
            return _Response(status=200, headers=self._identity_headers(), url=self.url)

        assert method == "GET"
        range_header = request.get_header("Range")
        assert range_header is not None
        start, end = self._parse_range(range_header)
        self.get_requests.append(
            {
                "range": range_header,
                "if_match": self._request_header(request, "If-Match"),
                "if_range": self._request_header(request, "If-Range"),
                "accept_encoding": self._request_header(request, "Accept-Encoding"),
            }
        )
        if self.transient_failures_remaining:
            self.transient_failures_remaining -= 1
            raise URLError("temporary test transport failure")

        body = self.data[start : end + 1]
        if self.body_transform is not None:
            body = self.body_transform(body, start, end)
        status = self.status_override if self.status_override is not None else 206
        content_range = self.content_range_override or f"bytes {start}-{end}/{len(self.data)}"
        headers = self._identity_headers(etag=self.response_etags.get(start, self.etag))
        headers["Content-Length"] = str(len(body))
        headers["Content-Range"] = content_range

        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)

        def release() -> None:
            with self._lock:
                self._active -= 1

        return _Response(
            status=status,
            headers=headers,
            body=body,
            url=self.url,
            read_delay=self.read_delay,
            on_close=release,
        )

    def _identity_headers(self, *, etag: str | None = None) -> dict[str, str]:
        headers = {"Content-Length": str(len(self.data))}
        current_etag = self.etag if etag is None else etag
        if current_etag is not None:
            headers["ETag"] = current_etag
        if self.last_modified is not None:
            headers["Last-Modified"] = self.last_modified
        return headers

    @staticmethod
    def _parse_range(value: str) -> tuple[int, int]:
        prefix, range_value = value.split("=", 1)
        assert prefix == "bytes"
        start, end = range_value.split("-", 1)
        return int(start), int(end)

    @staticmethod
    def _request_header(request, name: str) -> str | None:
        for key, value in request.header_items():
            if key.lower() == name.lower():
                return value
        return None


def _source(server: _RangeServer) -> itch_transport.SourceIdentity:
    return itch_transport.probe_source(server.url, opener=server, retries=0)


def _stream(
    server: _RangeServer,
    cache_dir: Path,
    *,
    source: itch_transport.SourceIdentity | None = None,
    chunk_bytes: int = 4,
    workers: int = 3,
    retries: int = 0,
) -> list[bytes]:
    return list(
        itch_transport.iter_cached_ranges(
            server.url,
            cache_dir,
            source=source,
            chunk_bytes=chunk_bytes,
            workers=workers,
            retries=retries,
            opener=server,
            log=None,
        )
    )


def test_ordered_chunks_use_bounded_parallel_requests(tmp_path: Path):
    server = _RangeServer(b"abcdefghijkl")
    server.read_delay = 0.03
    source = _source(server)

    chunks = _stream(server, tmp_path, source=source, workers=3)

    assert chunks == [b"abcd", b"efgh", b"ijkl"]
    assert server.max_active == 3
    assert [request["range"] for request in server.get_requests] == [
        "bytes=0-3",
        "bytes=4-7",
        "bytes=8-11",
    ]
    assert all(request["if_match"] == '"version-1"' for request in server.get_requests)
    assert all(request["accept_encoding"] == "identity" for request in server.get_requests)


def test_worker_count_is_capped_at_four_requests(tmp_path: Path):
    server = _RangeServer(b"abcdefghijklmnopqrst")
    server.read_delay = 0.03
    source = _source(server)

    assert b"".join(_stream(server, tmp_path, source=source, workers=99)) == server.data
    assert server.max_active == itch_transport.MAX_CONCURRENT_RANGES


def test_valid_cache_resumes_without_refetching(tmp_path: Path):
    server = _RangeServer(b"abcdefghij")
    source = _source(server)

    assert b"".join(_stream(server, tmp_path, source=source)) == server.data
    fetched = len(server.get_requests)
    assert b"".join(_stream(server, tmp_path, source=source)) == server.data

    assert len(server.get_requests) == fetched
    cache_path = itch_transport.cache_path_for(tmp_path, source)
    assert (cache_path / "source.json").is_file()
    assert len(list((cache_path / "parts").glob("*.part"))) == 3


def test_corrupt_cached_part_is_verified_then_repaired(tmp_path: Path):
    server = _RangeServer(b"abcdefghijkl")
    source = _source(server)
    assert b"".join(_stream(server, tmp_path, source=source)) == server.data
    fetched = len(server.get_requests)

    damaged = next((itch_transport.cache_path_for(tmp_path, source) / "parts").glob("*.part"))
    damaged.write_bytes(b"Z" * damaged.stat().st_size)

    assert b"".join(_stream(server, tmp_path, source=source)) == server.data
    assert len(server.get_requests) == fetched + 1


def test_identity_change_uses_a_separate_cache(tmp_path: Path):
    server = _RangeServer(b"abcdefgh")
    first = _source(server)
    assert b"".join(_stream(server, tmp_path, source=first)) == b"abcdefgh"
    fetched = len(server.get_requests)

    server.data = b"ABCDEFGH"
    server.etag = '"version-2"'
    second = _source(server)
    assert itch_transport.cache_path_for(tmp_path, first) != itch_transport.cache_path_for(tmp_path, second)

    assert b"".join(_stream(server, tmp_path, source=second)) == b"ABCDEFGH"
    assert len(server.get_requests) == fetched + 2


@pytest.mark.parametrize("failure", ["ignored", "wrong_content_range", "truncated"])
def test_invalid_range_response_fails_without_retry(tmp_path: Path, failure: str):
    server = _RangeServer(b"abcdefgh")
    if failure == "ignored":
        server.status_override = 200
    elif failure == "wrong_content_range":
        server.content_range_override = "bytes 1-4/8"
    else:
        server.body_transform = lambda body, _start, _end: body[:-1]
    source = _source(server)

    with pytest.raises(itch_transport.RangeResponseError):
        _stream(server, tmp_path, source=source, retries=2)

    assert len(server.get_requests) == 1


def test_transient_failures_retry_a_bounded_number_of_times(tmp_path: Path, monkeypatch):
    server = _RangeServer(b"abcd")
    server.transient_failures_remaining = 3
    source = _source(server)
    monkeypatch.setattr(itch_transport.time, "sleep", lambda _delay: None)

    with pytest.raises(itch_transport.RetryExhaustedError):
        _stream(server, tmp_path, source=source, retries=2)

    assert len(server.get_requests) == 3


@pytest.mark.parametrize("status", [400, 404])
def test_hard_http_statuses_are_not_retried(tmp_path: Path, status: int):
    server = _RangeServer(b"abcd")
    server.status_override = status
    source = _source(server)

    with pytest.raises(itch_transport.TransportError):
        _stream(server, tmp_path, source=source, retries=2)

    assert len(server.get_requests) == 1


def test_final_short_range_and_last_modified_fallback(tmp_path: Path):
    server = _RangeServer(b"abcdefghij", etag=None)
    source = _source(server)

    chunks = _stream(server, tmp_path, source=source)

    assert chunks == [b"abcd", b"efgh", b"ij"]
    assert [request["range"] for request in server.get_requests] == [
        "bytes=0-3",
        "bytes=4-7",
        "bytes=8-9",
    ]
    assert all(request["if_match"] is None for request in server.get_requests)
    assert all(request["if_range"] == server.last_modified for request in server.get_requests)


def test_mid_download_etag_change_stops_the_stream(tmp_path: Path):
    server = _RangeServer(b"abcdefgh")
    source = _source(server)
    server.response_etags[4] = '"version-2"'

    with pytest.raises(itch_transport.SourceChangedError):
        _stream(server, tmp_path, source=source, workers=1)

    assert len(server.get_requests) == 2


def test_cleanup_only_removes_the_identity_specific_cache(tmp_path: Path):
    server = _RangeServer(b"abcd")
    source = _source(server)
    _stream(server, tmp_path, source=source)
    cache_path = itch_transport.cache_path_for(tmp_path, source)

    assert itch_transport.cleanup_cached_ranges(tmp_path, source) == cache_path
    assert not cache_path.exists()
    assert (tmp_path / itch_transport.CACHE_DIRECTORY_NAME).is_dir()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"workers": 0}, "workers"),
        ({"chunk_bytes": 0}, "chunk_bytes"),
        ({"timeout": 0}, "timeout"),
        ({"retries": -1}, "retries"),
    ],
)
def test_range_controls_are_validated(tmp_path: Path, kwargs: dict[str, int], message: str):
    server = _RangeServer(b"abcd")
    source = _source(server)
    options = {"source": source, "opener": server, "log": None}
    options.update(kwargs)

    with pytest.raises(ValueError, match=message):
        list(itch_transport.iter_cached_ranges(server.url, tmp_path, **options))

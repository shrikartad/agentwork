"""Bounded, resumable HTTP Range transport for public ITCH gzip files.

``iter_cached_ranges`` downloads a pinned source in ordered compressed chunks while
keeping at most four requests in flight.  Each successful range is atomically
written to an identity-keyed cache and its SHA-256 is verified before reuse.
The cache is deliberately retained after iteration: remove it only after the
caller has validated the complete gzip and written its symbol manifest.

The module uses only the Python standard library.  ``opener`` is an optional
``urllib.request.urlopen``-compatible seam for offline tests.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CACHE_SCHEMA_VERSION = 1
CACHE_DIRECTORY_NAME = ".itch_range_cache_v1"
DEFAULT_CHUNK_BYTES = 8 << 20
DEFAULT_RETRIES = 3
DEFAULT_TIMEOUT = 120.0
MAX_CONCURRENT_RANGES = 4
MAX_CHUNK_BYTES = 64 << 20
MAX_CONTENT_LENGTH = (1 << 63) - 1
MAX_RETRIES = 10
MAX_TIMEOUT = 3600.0
_READ_BLOCK_BYTES = 1 << 20
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CONTENT_RANGE_RE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+)", re.IGNORECASE)

LogFn = Callable[[str], None]
OpenFn = Callable[..., Any]


class TransportError(RuntimeError):
    """Base error for a source or cache that cannot be safely consumed."""


class SourceProbeError(TransportError):
    """The source's HEAD response cannot establish a stable identity."""


class SourceChangedError(TransportError):
    """A Range response no longer matches the source identity from HEAD."""


class RangeResponseError(TransportError):
    """A server did not honor or correctly fulfill a requested byte range."""


class RetryExhaustedError(TransportError):
    """A transient request failed after its configured number of retries."""


class _StatusError(TransportError):
    def __init__(self, status: int, context: str) -> None:
        self.status = status
        super().__init__(f"{context} returned HTTP {status}")


class _RequestDeadlineExceeded(TransportError):
    pass


@dataclass(frozen=True)
class SourceIdentity:
    """Immutable source metadata pinned by a successful HEAD request."""

    url: str
    resolved_url: str
    content_length: int
    etag: str | None
    last_modified: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.url, str) or not self.url:
            raise ValueError("source URL must be a non-empty string")
        if not isinstance(self.resolved_url, str) or not self.resolved_url:
            raise ValueError("resolved source URL must be a non-empty string")
        if (
            isinstance(self.content_length, bool)
            or not isinstance(self.content_length, int)
            or not 0 < self.content_length <= MAX_CONTENT_LENGTH
        ):
            raise ValueError("source content length must be a positive signed-64-bit integer")
        if self.etag is not None and (not isinstance(self.etag, str) or not self.etag):
            raise ValueError("ETag must be a non-empty string when supplied")
        if self.last_modified is not None and (
            not isinstance(self.last_modified, str) or not self.last_modified
        ):
            raise ValueError("Last-Modified must be a non-empty string when supplied")
        if self.etag is None and self.last_modified is None:
            raise ValueError("source identity requires an ETag or Last-Modified validator")

    def as_dict(self) -> dict[str, object]:
        return {
            "url": self.url,
            "resolved_url": self.resolved_url,
            "content_length": self.content_length,
            "etag": self.etag,
            "last_modified": self.last_modified,
        }

    @property
    def cache_key(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class _Part:
    index: int
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start + 1


def probe_source(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    log: LogFn | None = None,
    opener: OpenFn | None = None,
) -> SourceIdentity:
    """Pin ``url`` to its length and ETag or Last-Modified HEAD metadata.

    ``retries`` is the number of extra attempts after the first attempt.  Only
    network failures and transient HTTP statuses are retried.
    """
    _validate_url(url)
    checked_timeout = _validate_timeout(timeout)
    checked_retries = _validate_retries(retries)
    open_request = opener or urlopen
    return _retry(
        lambda: _probe_source_once(url, checked_timeout, open_request),
        retries=checked_retries,
        log=log,
        label="HEAD request",
    )


def cache_path_for(cache_dir: Path, source: SourceIdentity) -> Path:
    """Return the private cache directory for one exact source identity."""
    _validate_source(source)
    url_key = hashlib.sha256(source.url.encode("utf-8")).hexdigest()
    return Path(cache_dir) / CACHE_DIRECTORY_NAME / url_key / source.cache_key


def cleanup_cached_ranges(cache_dir: Path, source: SourceIdentity) -> Path:
    """Remove one identity-keyed cache after complete gzip/manifest validation.

    The operation can remove only the cache path derived from ``source`` under
    ``cache_dir``; it never recursively removes the caller's cache root.
    """
    target = cache_path_for(cache_dir, source)
    root = (Path(cache_dir) / CACHE_DIRECTORY_NAME).resolve()
    resolved_target = target.resolve()
    if resolved_target.parent.parent != root:
        raise TransportError("refusing to remove a cache path outside the transport cache root")
    if resolved_target.exists():
        shutil.rmtree(resolved_target)
    try:
        resolved_target.parent.rmdir()
    except OSError:
        pass
    return target


def iter_cached_ranges(
    url: str,
    cache_dir: Path,
    *,
    source: SourceIdentity | None = None,
    workers: int = MAX_CONCURRENT_RANGES,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    log: LogFn | None = print,
    opener: OpenFn | None = None,
) -> Iterator[bytes]:
    """Yield a pinned source as ordered compressed Range chunks.

    At most four requests are active even if a larger ``workers`` value is
    supplied.  Parts are cached below ``cache_dir`` and remain there after the
    iterator completes or fails, so a later invocation resumes safely.  Call
    :func:`cleanup_cached_ranges` only after the caller validates the complete
    gzip stream and writes its verified output manifest.
    """
    _validate_url(url)
    checked_workers = _validate_workers(workers, log)
    checked_chunk_bytes = _validate_chunk_bytes(chunk_bytes)
    checked_timeout = _validate_timeout(timeout)
    checked_retries = _validate_retries(retries)
    open_request = opener or urlopen

    if source is None:
        source = probe_source(
            url,
            timeout=checked_timeout,
            retries=checked_retries,
            log=log,
            opener=open_request,
        )
    _validate_source(source)
    if source.url != url:
        raise ValueError("provided source identity belongs to a different URL")

    cache_path = cache_path_for(cache_dir, source)
    parts_dir = _prepare_cache(cache_path, source)
    part_count = (source.content_length + checked_chunk_bytes - 1) // checked_chunk_bytes

    executor = ThreadPoolExecutor(max_workers=checked_workers, thread_name_prefix="itch-range")
    scheduled: dict[int, Future[bytes] | None] = {}
    next_to_schedule = 0
    next_to_yield = 0

    def schedule_window() -> None:
        nonlocal next_to_schedule
        while (
            next_to_schedule < part_count
            and next_to_schedule - next_to_yield < checked_workers
        ):
            part = _part_at(next_to_schedule, source.content_length, checked_chunk_bytes)
            if _is_cached_part_candidate(parts_dir, source, part):
                scheduled[part.index] = None
            else:
                scheduled[part.index] = executor.submit(
                    _fetch_and_store_part,
                    source,
                    part,
                    parts_dir,
                    checked_timeout,
                    checked_retries,
                    log,
                    open_request,
                )
            next_to_schedule += 1

    try:
        schedule_window()
        while next_to_yield < part_count:
            part = _part_at(next_to_yield, source.content_length, checked_chunk_bytes)
            future = scheduled.pop(part.index)
            if future is None:
                data = _read_cached_part(parts_dir, source, part)
                if data is None:
                    _emit(log, f"range cache checksum mismatch at bytes {part.start}-{part.end}; refetching")
                    data = executor.submit(
                        _fetch_and_store_part,
                        source,
                        part,
                        parts_dir,
                        checked_timeout,
                        checked_retries,
                        log,
                        open_request,
                    ).result()
            else:
                data = future.result()

            if len(data) != part.size:
                raise RangeResponseError(
                    f"cached bytes {part.start}-{part.end} have length {len(data)}, expected {part.size}"
                )
            yield data
            next_to_yield += 1
            schedule_window()
    finally:
        for future in scheduled.values():
            if future is not None:
                future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)


def _probe_source_once(url: str, timeout: float, opener: OpenFn) -> SourceIdentity:
    response: Any | None = None
    try:
        request = Request(url, method="HEAD", headers=_base_headers())
        response = opener(request, timeout=timeout)
        status = _response_status(response)
        if status != 200:
            raise _StatusError(status, "HEAD request")
        content_length = _parse_uint_header(_header(response, "Content-Length"), "Content-Length")
        if content_length == 0:
            raise SourceProbeError("HEAD Content-Length must be greater than zero")
        if content_length > MAX_CONTENT_LENGTH:
            raise SourceProbeError("HEAD Content-Length exceeds signed-64-bit range")
        etag = _header(response, "ETag")
        last_modified = _header(response, "Last-Modified")
        if etag is None and last_modified is None:
            raise SourceProbeError("HEAD response needs an ETag or Last-Modified validator")
        resolved_url = _response_url(response, url)
        return SourceIdentity(
            url=url,
            resolved_url=resolved_url,
            content_length=content_length,
            etag=etag,
            last_modified=last_modified,
        )
    except HTTPError as exc:
        raise _StatusError(exc.code, "HEAD request") from exc
    finally:
        _close_response(response)


def _fetch_and_store_part(
    source: SourceIdentity,
    part: _Part,
    parts_dir: Path,
    timeout: float,
    retries: int,
    log: LogFn | None,
    opener: OpenFn,
) -> bytes:
    data = _retry(
        lambda: _fetch_part_once(source, part, timeout, opener),
        retries=retries,
        log=log,
        label=f"Range bytes {part.start}-{part.end}",
    )
    _store_part(parts_dir, source, part, data)
    return data


def _fetch_part_once(source: SourceIdentity, part: _Part, timeout: float, opener: OpenFn) -> bytes:
    headers = _base_headers()
    headers["Range"] = f"bytes={part.start}-{part.end}"
    if source.etag and not source.etag.startswith("W/"):
        headers["If-Match"] = source.etag
    elif source.last_modified:
        headers["If-Range"] = source.last_modified

    response: Any | None = None
    deadline = time.monotonic() + timeout
    try:
        request = Request(source.url, headers=headers)
        response = opener(request, timeout=timeout)
        status = _response_status(response)
        if status == 200:
            _validate_response_identity(response, source)
            raise RangeResponseError(
                f"server ignored Range bytes {part.start}-{part.end} with HTTP 200"
            )
        if status != 206:
            raise _StatusError(status, f"Range bytes {part.start}-{part.end}")
        _validate_response_identity(response, source)
        _validate_content_range(response, source, part)
        declared_length = _header(response, "Content-Length")
        if declared_length is not None and _parse_uint_header(
            declared_length, "Range Content-Length"
        ) != part.size:
            raise RangeResponseError(
                f"Range bytes {part.start}-{part.end} declared {declared_length} bytes, expected {part.size}"
            )
        return _read_exact_range(response, part, deadline)
    except HTTPError as exc:
        if exc.code == 412:
            raise SourceChangedError("source rejected If-Match while fetching a cached range") from exc
        if exc.code == 416:
            raise RangeResponseError(
                f"server rejected Range bytes {part.start}-{part.end} with HTTP 416"
            ) from exc
        raise _StatusError(exc.code, f"Range bytes {part.start}-{part.end}") from exc
    finally:
        _close_response(response)


def _read_exact_range(response: Any, part: _Part, deadline: float) -> bytes:
    data = bytearray()
    while len(data) < part.size:
        remaining = _remaining_time(deadline)
        _set_response_timeout(response, remaining)
        block = response.read(min(_READ_BLOCK_BYTES, part.size - len(data)))
        if not block:
            break
        if not isinstance(block, (bytes, bytearray, memoryview)):
            raise RangeResponseError("Range response yielded a non-bytes payload")
        data.extend(block)
        if len(data) > part.size:
            raise RangeResponseError(
                f"Range bytes {part.start}-{part.end} returned more than {part.size} bytes"
            )

    if len(data) != part.size:
        raise RangeResponseError(
            f"Range bytes {part.start}-{part.end} was truncated at {len(data)} of {part.size} bytes"
        )

    remaining = _remaining_time(deadline)
    _set_response_timeout(response, remaining)
    extra = response.read(1)
    if extra:
        raise RangeResponseError(
            f"Range bytes {part.start}-{part.end} returned more than {part.size} bytes"
        )
    return bytes(data)


def _validate_response_identity(response: Any, source: SourceIdentity) -> None:
    if source.etag is not None:
        got_etag = _header(response, "ETag")
        if got_etag != source.etag:
            raise SourceChangedError(
                f"Range response ETag {got_etag!r} does not match HEAD ETag {source.etag!r}"
            )
    elif source.last_modified is not None:
        got_last_modified = _header(response, "Last-Modified")
        if got_last_modified != source.last_modified:
            raise SourceChangedError("Range response Last-Modified does not match the HEAD response")


def _validate_content_range(response: Any, source: SourceIdentity, part: _Part) -> None:
    raw = _header(response, "Content-Range")
    match = _CONTENT_RANGE_RE.fullmatch(raw or "")
    if match is None:
        raise RangeResponseError(f"Range bytes {part.start}-{part.end} has invalid Content-Range {raw!r}")
    start, end, total = (int(value) for value in match.groups())
    if (start, end, total) != (part.start, part.end, source.content_length):
        raise RangeResponseError(
            "Range response Content-Range "
            f"bytes {start}-{end}/{total} does not match bytes {part.start}-{part.end}/{source.content_length}"
        )


def _prepare_cache(cache_path: Path, source: SourceIdentity) -> Path:
    parts_dir = cache_path / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_path / "source.json"
    expected = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "identity_key": source.cache_key,
        "source": source.as_dict(),
    }
    if _read_json(manifest_path) != expected:
        _atomic_write_json(manifest_path, expected)
    return parts_dir


def _part_at(index: int, content_length: int, chunk_bytes: int) -> _Part:
    start = index * chunk_bytes
    if start < 0 or start >= content_length:
        raise ValueError("part index is outside the pinned source length")
    end = min(start + chunk_bytes, content_length) - 1
    return _Part(index=index, start=start, end=end)


def _part_paths(parts_dir: Path, part: _Part) -> tuple[Path, Path]:
    stem = f"{part.start:020d}-{part.end:020d}"
    return parts_dir / f"{stem}.part", parts_dir / f"{stem}.json"


def _part_metadata(source: SourceIdentity, part: _Part, digest: str | None = None) -> dict[str, object]:
    metadata: dict[str, object] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "identity_key": source.cache_key,
        "start": part.start,
        "end": part.end,
        "bytes": part.size,
    }
    if digest is not None:
        metadata["sha256"] = digest
    return metadata


def _is_cached_part_candidate(parts_dir: Path, source: SourceIdentity, part: _Part) -> bool:
    data_path, metadata_path = _part_paths(parts_dir, part)
    metadata = _read_json(metadata_path)
    if metadata is None:
        return False
    expected = _part_metadata(source, part)
    if any(metadata.get(key) != value for key, value in expected.items()):
        return False
    digest = metadata.get("sha256")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        return False
    try:
        return data_path.is_file() and data_path.stat().st_size == part.size
    except OSError:
        return False


def _read_cached_part(parts_dir: Path, source: SourceIdentity, part: _Part) -> bytes | None:
    if not _is_cached_part_candidate(parts_dir, source, part):
        return None
    data_path, metadata_path = _part_paths(parts_dir, part)
    metadata = _read_json(metadata_path)
    if metadata is None:
        return None
    expected_digest = metadata.get("sha256")
    if not isinstance(expected_digest, str):
        return None
    try:
        data = data_path.read_bytes()
    except OSError:
        return None
    if len(data) != part.size:
        return None
    if hashlib.sha256(data).hexdigest() != expected_digest:
        return None
    return data


def _store_part(parts_dir: Path, source: SourceIdentity, part: _Part, data: bytes) -> None:
    if len(data) != part.size:
        raise RangeResponseError(
            f"refusing to cache bytes {part.start}-{part.end}: got {len(data)}, expected {part.size}"
        )
    data_path, metadata_path = _part_paths(parts_dir, part)
    digest = hashlib.sha256(data).hexdigest()
    _atomic_write_bytes(data_path, data)
    _atomic_write_json(metadata_path, _part_metadata(source, part, digest))


def _retry(
    operation: Callable[[], Any],
    *,
    retries: int,
    log: LogFn | None,
    label: str,
) -> Any:
    for attempt in range(retries + 1):
        try:
            return operation()
        except (HTTPError, URLError, OSError, TimeoutError, http.client.HTTPException) as exc:
            error: BaseException = exc
        except TransportError as exc:
            if not _is_retryable_transport_error(exc):
                raise
            error = exc
        if attempt == retries:
            raise RetryExhaustedError(f"{label} failed after {attempt + 1} attempt(s)") from error
        delay = min(0.25 * (2**attempt), 2.0)
        _emit(log, f"{label} failed ({error}); retrying in {delay:.2f}s")
        time.sleep(delay)
    raise AssertionError("retry loop must return or raise")


def _is_retryable_transport_error(error: TransportError) -> bool:
    if isinstance(error, _RequestDeadlineExceeded):
        return True
    if isinstance(error, _StatusError):
        return error.status in (408, 429) or 500 <= error.status <= 599
    return False


def _base_headers() -> dict[str, str]:
    return {
        "Accept-Encoding": "identity",
        "Cache-Control": "no-transform",
        "User-Agent": "nexus-lob-itch-transport/1.0",
    }


def _response_status(response: Any) -> int:
    status = getattr(response, "status", None)
    if status is None:
        getcode = getattr(response, "getcode", None)
        status = getcode() if callable(getcode) else None
    if isinstance(status, bool) or not isinstance(status, int):
        raise TransportError("HTTP response did not expose an integer status")
    return status


def _response_url(response: Any, fallback: str) -> str:
    geturl = getattr(response, "geturl", None)
    resolved = geturl() if callable(geturl) else fallback
    return resolved if isinstance(resolved, str) and resolved else fallback


def _header(response: Any, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    get = getattr(headers, "get", None)
    value = get(name) if callable(get) else None
    if value is None:
        items = getattr(headers, "items", None)
        if callable(items):
            for key, candidate in items():
                if str(key).lower() == name.lower():
                    value = candidate
                    break
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _parse_uint_header(value: str | None, name: str) -> int:
    if value is None or not value.isascii() or not value.isdecimal():
        raise SourceProbeError(f"{name} must be an unsigned decimal integer")
    parsed = int(value)
    if parsed < 0 or parsed > MAX_CONTENT_LENGTH:
        raise SourceProbeError(f"{name} is outside the signed-64-bit range")
    return parsed


def _remaining_time(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _RequestDeadlineExceeded("Range request exceeded its total timeout")
    return remaining


def _set_response_timeout(response: Any, timeout: float) -> None:
    fp = getattr(response, "fp", None)
    raw = getattr(fp, "raw", None)
    sock = getattr(raw, "_sock", None)
    settimeout = getattr(sock, "settimeout", None)
    if callable(settimeout):
        try:
            settimeout(timeout)
        except OSError:
            pass


def _close_response(response: Any | None) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        try:
            close()
        except OSError:
            pass


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
    )


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _validate_url(url: str) -> None:
    if not isinstance(url, str) or not url:
        raise ValueError("url must be a non-empty string")


def _validate_source(source: SourceIdentity) -> None:
    if not isinstance(source, SourceIdentity):
        raise TypeError("source must be a SourceIdentity")


def _validate_workers(workers: int, log: LogFn | None) -> int:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    bounded = min(workers, MAX_CONCURRENT_RANGES)
    if bounded != workers:
        _emit(log, f"workers={workers} capped at {MAX_CONCURRENT_RANGES} concurrent Range requests")
    return bounded


def _validate_chunk_bytes(chunk_bytes: int) -> int:
    if (
        isinstance(chunk_bytes, bool)
        or not isinstance(chunk_bytes, int)
        or not 0 < chunk_bytes <= MAX_CHUNK_BYTES
    ):
        raise ValueError(f"chunk_bytes must be an integer between 1 and {MAX_CHUNK_BYTES}")
    return chunk_bytes


def _validate_timeout(timeout: float) -> float:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError("timeout must be a positive number")
    checked = float(timeout)
    if not math.isfinite(checked) or not 0 < checked <= MAX_TIMEOUT:
        raise ValueError(f"timeout must be between 0 and {MAX_TIMEOUT:g} seconds")
    return checked


def _validate_retries(retries: int) -> int:
    if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= MAX_RETRIES:
        raise ValueError(f"retries must be an integer between 0 and {MAX_RETRIES}")
    return retries


def _emit(log: LogFn | None, message: str) -> None:
    if log is not None:
        log(message)

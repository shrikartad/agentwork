"""Injectable book surface shared by replay and OrderBookEnv.

Person A's C++ ``nexus_engine.Engine`` and Person B's ``StubOrderBook`` do
not share a mutation API:

* Stub: ``add(side, price, size)`` / ``cancel(side, price, size)`` / ``record_trade``
* Engine: ``submit_limit`` / ``submit_market`` / ``cancel(order_id)`` / ``fills``

Both expose the frozen ``view()`` / ``snapshot()`` contract. This module is
the Python-side adapter so replay/env never import a concrete engine class.
Swap the adapter, not the env.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .book_state import Side, StubOrderBook

View = Mapping[str, Any]


@runtime_checkable
class BookView(Protocol):
    """Frozen observation seam — valid for StubOrderBook and Engine."""

    def view(self) -> View:
        """Live window. Engine arrays alias engine memory until the next mutation."""

    def snapshot(self) -> View:
        """Owning copy, safe to retain across steps."""


@dataclass(slots=True)
class Resting:
    order_id: int
    side: Side
    price: int
    size: int


@dataclass(slots=True)
class TakeResult:
    filled: int
    notional_ticks: int

    @property
    def avg_px(self) -> int:
        return 0 if self.filled == 0 else int(self.notional_ticks // self.filled)


@runtime_checkable
class ExecutableBook(BookView, Protocol):
    """Mutations the env needs. Implemented by adapters, not by the stub itself."""

    def take(self, side: Side, qty: int, limit_px: int | None = None) -> TakeResult:
        """Aggressive walk of the opposite book. ``side`` is the *taker* side."""

    def rest(self, side: Side, price: int, qty: int, order_id: int | None = None) -> Resting:
        """Post a resting limit. Returns the handle used for later cancel."""

    def cancel_resting(self, handle: Resting) -> int:
        """Cancel residual at ``handle``. Returns cancelled size."""

    def record_trade(self, side: Side, price: int, size: int) -> None: ...

    def reset(self) -> None:
        """Return the adapter to an empty book for a new environment episode."""


class StubBookAdapter:
    """Replay + env adapter over the frozen ``StubOrderBook``.

    Order-id state lives here; the stub only sees aggregate
    ``add`` / ``cancel`` / ``record_trade`` calls and in-place reset.
    """

    def __init__(self, book: StubOrderBook | None = None) -> None:
        self.book = book if book is not None else StubOrderBook()
        self._orders: dict[int, Resting] = {}
        self._next_id = 1

    def view(self) -> View:
        return self.book.view()

    def snapshot(self) -> View:
        v = self.book.snapshot()
        return {k: (np.copy(x) if isinstance(x, np.ndarray) else x) for k, x in v.items()}

    def rest(self, side: Side, price: int, qty: int, order_id: int | None = None) -> Resting:
        oid = int(order_id) if order_id is not None else self._next_id
        self._next_id = max(self._next_id, oid + 1)
        if int(qty) <= 0 or oid in self._orders:
            return Resting(oid, side, int(price), 0)
        self.book.add(side, int(price), int(qty), orders=1)
        h = Resting(oid, side, int(price), int(qty))
        self._orders[oid] = h
        return h

    def cancel_resting(self, handle: Resting) -> int:
        live = self._orders.get(handle.order_id)
        if live is None or live.size <= 0:
            return 0
        cut = live.size
        self.book.cancel(live.side, live.price, cut, orders=1)
        live.size = 0
        self._orders.pop(handle.order_id, None)
        return cut

    def cancel_id(self, order_id: int, size: int | None = None) -> int:
        live = self._orders.get(int(order_id))
        if live is None:
            return 0
        cut = live.size if size is None else min(live.size, int(size))
        if cut <= 0:
            return 0
        orders = 1 if cut >= live.size else 0
        self.book.cancel(live.side, live.price, cut, orders=orders)
        live.size -= cut
        if live.size <= 0:
            self._orders.pop(live.order_id, None)
        return cut

    def lookup(self, order_id: int) -> Resting | None:
        return self._orders.get(int(order_id))

    def record_trade(self, side: Side, price: int, size: int) -> None:
        self.book.record_trade(side, int(price), int(size))

    def take(self, side: Side, qty: int, limit_px: int | None = None) -> TakeResult:
        """Walk opposite displayed levels. Does not require C++ matching."""
        remaining = int(qty)
        notional = 0
        if side == Side.Bid:
            opp = Side.Ask
        else:
            opp = Side.Bid
        for px, sz in self._levels(opp):
            if px == 0 or sz <= 0 or remaining <= 0:
                continue
            if limit_px is not None:
                if side == Side.Bid and px > limit_px:
                    break
                if side == Side.Ask and px < limit_px:
                    break
            hit = min(remaining, sz)
            completed = self._hit_orders(opp, px, hit)
            self.book.cancel(opp, px, hit, orders=completed)
            self.book.record_trade(side, px, hit)
            remaining -= hit
            notional += hit * px
        return TakeResult(filled=int(qty) - remaining, notional_ticks=notional)

    def _levels(self, side: Side) -> list[tuple[int, int]]:
        """Return all stub levels, not just the ABI-visible top ``DEPTH`` levels."""
        raw = getattr(self.book, "_bids" if side == Side.Bid else "_asks", None)
        if isinstance(raw, dict):
            return sorted(
                ((int(price), int(level[0])) for price, level in raw.items()),
                key=lambda level: level[0],
                reverse=side == Side.Bid,
            )

        state = self.snapshot()
        px_key = "bid_px" if side == Side.Bid else "ask_px"
        sz_key = "bid_sz" if side == Side.Bid else "ask_sz"
        return [(int(price), int(size)) for price, size in zip(state[px_key], state[sz_key])]

    def _hit_orders(self, side: Side, price: int, qty: int) -> int:
        """Reduce tracked makers and return how many were fully consumed."""
        left = qty
        completed = 0
        for oid, live in list(self._orders.items()):
            if left <= 0:
                break
            if live.side != side or live.price != price or live.size <= 0:
                continue
            take = min(live.size, left)
            live.size -= take
            left -= take
            if live.size <= 0:
                self._orders.pop(oid, None)
                completed += 1
        return completed

    def reset(self) -> None:
        """Clear liquidity in-place so an injected StubOrderBook stays the same object."""
        if hasattr(self.book, "reset"):
            self.book.reset()
        else:
            snap = self.snapshot()
            for side, px_key, sz_key, ct_key in (
                (Side.Bid, "bid_px", "bid_sz", "bid_ct"),
                (Side.Ask, "ask_px", "ask_sz", "ask_ct"),
            ):
                for price, size, count in zip(snap[px_key], snap[sz_key], snap[ct_key]):
                    if int(size):
                        self.book.cancel(side, int(price), int(size), orders=int(count))
        self._orders.clear()
        self._next_id = 1


class EngineAdapter:
    """Thin wrapper around a compiled ``nexus_engine.Engine``.

    Import of ``nexus_engine`` is deferred so the package imports without a
    C++ build. Construct with ``EngineAdapter(nexus_engine.Engine())``.
    """

    def __init__(
        self,
        engine: Any,
        *,
        engine_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.engine = engine
        self._engine_factory = engine_factory or type(engine)
        self._next_id = 1
        self._live: dict[int, Resting] = {}

    def reset(self) -> None:
        """Replace the opaque engine with a fresh empty instance for a new episode."""
        self.engine = self._engine_factory()
        self._live.clear()
        self._next_id = 1

    def view(self) -> View:
        return self.engine.view()

    def snapshot(self) -> View:
        return self.engine.snapshot()

    def rest(self, side: Side, price: int, qty: int, order_id: int | None = None) -> Resting:
        oid = int(order_id) if order_id is not None else self._next_id
        self._next_id = max(self._next_id, oid + 1)
        tif = _engine_tif(self.engine, "GTC")
        eng_side = _engine_side(self.engine, side)
        result = self.engine.submit_limit(oid, eng_side, int(price), int(qty), tif)
        # ``resting`` is authoritative: fills are per-call diagnostics and are
        # empty for rejections such as a duplicate id or an out-of-band price.
        live_sz = _result_int(result, "resting")
        fills = list(self.engine.fills()) if hasattr(self.engine, "fills") else []
        if live_sz is None:
            live_sz = int(qty) - sum(int(fill[3]) for fill in fills)
        live_sz = max(0, min(int(qty), live_sz))
        self._consume_maker_fills(fills)
        h = Resting(oid, side, int(price), live_sz)
        if live_sz > 0:
            self._live[oid] = h
        return h

    def cancel_resting(self, handle: Resting) -> int:
        oid = int(handle.order_id)
        live = self._live.get(oid)
        if live is None or live.size <= 0:
            return 0
        result = self.engine.cancel(oid)
        status = _result_status(result)
        if status == "NoOp":
            live.size = 0
            self._live.pop(oid, None)
            return 0
        if status is not None and status.startswith("Rejected"):
            return 0
        cancelled = live.size
        live.size = 0
        self._live.pop(oid, None)
        return cancelled

    def lookup(self, order_id: int) -> Resting | None:
        """Live resting handle for ``order_id``, or None if not resting."""
        return self._live.get(int(order_id))

    def cancel_id(self, order_id: int, size: int | None = None) -> int:
        """Cancel ``size`` (default: the whole order) of a resting order.

        Mirrors ``StubBookAdapter.cancel_id`` so ``ReplayEngine.apply`` drives the
        real engine the same way it drives the stub. A full residual uses
        ``engine.cancel``; a partial reduction uses ``engine.modify`` at the same
        price, which keeps time priority (matches the stub's partial-cancel
        semantics). Returns the number of shares actually cancelled.
        """
        live = self._live.get(int(order_id))
        if live is None or live.size <= 0:
            return 0
        cut = live.size if size is None else min(live.size, int(size))
        if cut <= 0:
            return 0
        previous_size = live.size
        remaining = live.size - cut
        if remaining <= 0:
            result = self.engine.cancel(int(order_id))
            status = _result_status(result)
            if status == "NoOp":
                live.size = 0
                self._live.pop(int(order_id), None)
                return 0
            if status is not None and status.startswith("Rejected"):
                return 0
            live.size = 0
            self._live.pop(int(order_id), None)
        else:
            # In-place size reduction keeps the order resting with time priority.
            result = self.engine.modify(int(order_id), int(live.price), remaining)
            status = _result_status(result)
            if status == "NoOp":
                live.size = 0
                self._live.pop(int(order_id), None)
                return 0
            if status is not None and status.startswith("Rejected"):
                return 0
            live_sz = _result_int(result, "resting")
            if live_sz is None:
                live_sz = remaining
            live_sz = max(0, min(remaining, live_sz))
            live.size = live_sz
            if live_sz <= 0:
                self._live.pop(int(order_id), None)
            return previous_size - live_sz
        return cut

    def record_trade(self, side: Side, price: int, size: int) -> None:
        # Engine records prints on matching; explicit tape prints are a no-op here.
        del side, price, size

    def take(self, side: Side, qty: int, limit_px: int | None = None) -> TakeResult:
        oid = self._next_id
        self._next_id += 1
        eng_side = _engine_side(self.engine, side)
        if limit_px is None:
            r = self.engine.submit_market(oid, eng_side, int(qty))
        else:
            tif = _engine_tif(self.engine, "IOC")
            r = self.engine.submit_limit(oid, eng_side, int(limit_px), int(qty), tif)
        filled = int(r.get("filled", 0)) if isinstance(r, dict) else int(getattr(r, "filled", 0))
        notional = 0
        fills = list(self.engine.fills()) if hasattr(self.engine, "fills") else []
        for f in fills:
            # (maker, taker, px, qty, aggressor)
            notional += int(f[2]) * int(f[3])
        if filled and notional == 0:
            px = int(self.view().get("last_trade_px") or 0)
            notional = filled * px
        self._consume_maker_fills(fills)
        return TakeResult(filled=filled, notional_ticks=notional)

    def _consume_maker_fills(self, fills: list[Any]) -> None:
        """Reconcile adapter handles after an engine call crosses resting liquidity."""
        for f in fills:
            maker = int(f[0])
            qty = int(f[3])
            live = self._live.get(maker)
            if live is None:
                continue
            live.size = max(0, live.size - qty)
            if live.size <= 0:
                self._live.pop(maker, None)


def _result_int(result: Any, key: str) -> int | None:
    value = result.get(key) if isinstance(result, Mapping) else getattr(result, key, None)
    return None if value is None else int(value)


def _result_status(result: Any) -> str | None:
    value = result.get("status") if isinstance(result, Mapping) else getattr(result, "status", None)
    if value is None:
        return None
    return str(getattr(value, "name", value)).rsplit(".", maxsplit=1)[-1]


def _engine_enum(engine: Any, name: str) -> Any | None:
    enum = getattr(type(engine), name, None) or getattr(engine, name, None)
    if enum is None:
        try:
            import nexus_engine as ne

            enum = getattr(ne, name)
        except ImportError:
            return None
    return enum


def _engine_side(engine: Any, side: Side) -> Any:
    enum = _engine_enum(engine, "Side")
    if enum is None:
        return int(side)
    return enum.Bid if side == Side.Bid else enum.Ask


def _engine_tif(engine: Any, name: str) -> Any:
    enum = _engine_enum(engine, "TimeInForce")
    return getattr(enum, name) if enum is not None else name


def adapt(book: Any) -> StubBookAdapter | EngineAdapter:
    """Wrap a stub or a compiled Engine without the caller branching."""
    if isinstance(book, (StubBookAdapter, EngineAdapter)):
        return book
    if isinstance(book, StubOrderBook):
        return StubBookAdapter(book)
    if hasattr(book, "submit_limit") and hasattr(book, "view"):
        return EngineAdapter(book)
    if hasattr(book, "view") and hasattr(book, "add"):
        return StubBookAdapter(book)
    raise TypeError(f"cannot adapt book of type {type(book)!r}")

"""Adapter parity tests for the stub, a strict pybind-shaped fake, and Engine.

The unit cases always run. The compiled-engine cases exercise the real bridge
when it is available and skip individually when it is not.
"""
from __future__ import annotations

import sys
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import nexus_engine as ne
except ImportError:
    ne = None

from nexus_quant.baselines import run_episode
from nexus_quant.book_port import EngineAdapter, StubBookAdapter, adapt
from nexus_quant.book_state import (
    BOOK_STATE_DTYPE,
    CT_DTYPE,
    DEPTH,
    PX_DTYPE,
    SZ_DTYPE,
    Side,
    StubOrderBook,
)
from nexus_quant.envs.order_book_env import OBS_DIM, OrderBookEnv


class StrictEngine:
    """Small pybind-shaped test double; it validates the adapter, not matching parity."""

    class Side(IntEnum):
        Bid = 0
        Ask = 1

    class TimeInForce(IntEnum):
        GTC = 0
        IOC = 1

    def __init__(self) -> None:
        self._orders: dict[int, dict[str, Any]] = {}
        self._fills: list[tuple[int, int, int, int, StrictEngine.Side]] = []
        self._seq = 0
        self._cum_volume = 0
        self._last_trade_px = 0
        self._last_trade_sz = 0
        self._last_trade_side = self.Side.Ask
        self.modify_calls: list[tuple[int, int, int]] = []
        self.submit_limit_calls: list[tuple[int, StrictEngine.Side, int, int, StrictEngine.TimeInForce]] = []
        self._state: dict[str, Any] = {
            "bid_px": np.zeros(DEPTH, dtype=PX_DTYPE),
            "bid_sz": np.zeros(DEPTH, dtype=SZ_DTYPE),
            "bid_ct": np.zeros(DEPTH, dtype=CT_DTYPE),
            "ask_px": np.zeros(DEPTH, dtype=PX_DTYPE),
            "ask_sz": np.zeros(DEPTH, dtype=SZ_DTYPE),
            "ask_ct": np.zeros(DEPTH, dtype=CT_DTYPE),
            "seq": 0,
            "ts_ns": 0,
            "cum_volume": 0,
            "last_trade_px": 0,
            "last_trade_sz": 0,
            "last_trade_side": self.Side.Ask,
        }

    @staticmethod
    def _result(order_id: int, status: str, filled: int = 0, resting: int = 0) -> dict[str, int | str]:
        return {"id": order_id, "status": status, "filled": filled, "resting": resting}

    @staticmethod
    def _require_int(value: Any, name: str) -> int:
        assert type(value) is int, f"{name} must be a built-in int, got {type(value)!r}"
        return value

    def _validate_order(
        self,
        order_id: int,
        side: StrictEngine.Side,
        price: int,
        qty: int,
        tif: StrictEngine.TimeInForce,
    ) -> None:
        self._require_int(order_id, "order_id")
        self._require_int(price, "price")
        self._require_int(qty, "qty")
        assert isinstance(side, self.Side)
        assert isinstance(tif, self.TimeInForce)

    def _match(
        self,
        taker_id: int,
        taker_side: StrictEngine.Side,
        qty: int,
        limit_px: int | None,
    ) -> tuple[int, int]:
        maker_side = self.Side.Ask if taker_side == self.Side.Bid else self.Side.Bid
        candidates = [
            (order_id, order)
            for order_id, order in self._orders.items()
            if order["side"] == maker_side
        ]
        candidates.sort(
            key=lambda item: (
                -int(item[1]["price"])
                if maker_side == self.Side.Bid
                else int(item[1]["price"]),
                int(item[1]["seq"]),
            )
        )
        remaining = qty
        for maker_id, maker in candidates:
            if remaining <= 0:
                break
            price = int(maker["price"])
            if limit_px is not None and (
                (taker_side == self.Side.Bid and price > limit_px)
                or (taker_side == self.Side.Ask and price < limit_px)
            ):
                break
            filled = min(remaining, int(maker["size"]))
            maker["size"] = int(maker["size"]) - filled
            self._fills.append((maker_id, taker_id, price, filled, taker_side))
            self._cum_volume += filled
            self._last_trade_px = price
            self._last_trade_sz = filled
            self._last_trade_side = taker_side
            remaining -= filled
            if maker["size"] == 0:
                del self._orders[maker_id]
        return qty - remaining, remaining

    def _publish(self) -> None:
        for field in ("bid_px", "bid_sz", "bid_ct", "ask_px", "ask_sz", "ask_ct"):
            self._state[field].fill(0)
        for side, px_key, sz_key, ct_key, reverse in (
            (self.Side.Bid, "bid_px", "bid_sz", "bid_ct", True),
            (self.Side.Ask, "ask_px", "ask_sz", "ask_ct", False),
        ):
            levels: dict[int, list[dict[str, Any]]] = {}
            for order in self._orders.values():
                if order["side"] == side:
                    levels.setdefault(int(order["price"]), []).append(order)
            for index, (price, orders) in enumerate(sorted(levels.items(), reverse=reverse)[:DEPTH]):
                self._state[px_key][index] = price
                self._state[sz_key][index] = sum(int(order["size"]) for order in orders)
                self._state[ct_key][index] = len(orders)
        self._seq += 1
        self._state.update(
            seq=self._seq,
            ts_ns=self._seq,
            cum_volume=self._cum_volume,
            last_trade_px=self._last_trade_px,
            last_trade_sz=self._last_trade_sz,
            last_trade_side=self._last_trade_side,
        )

    def view(self) -> dict[str, Any]:
        return self._state

    def snapshot(self) -> dict[str, Any]:
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in self._state.items()
        }

    def fills(self) -> list[tuple[int, int, int, int, StrictEngine.Side]]:
        return list(self._fills)

    def submit_limit(
        self,
        order_id: int,
        side: StrictEngine.Side,
        price: int,
        qty: int,
        tif: StrictEngine.TimeInForce,
    ) -> dict[str, int | str]:
        self._validate_order(order_id, side, price, qty, tif)
        self.submit_limit_calls.append((order_id, side, price, qty, tif))
        self._fills.clear()
        if order_id in self._orders:
            return self._result(order_id, "Rejected_DupId")
        if qty <= 0:
            return self._result(order_id, "Rejected_BadQty")
        filled, remaining = self._match(order_id, side, qty, price)
        if remaining and tif == self.TimeInForce.GTC:
            self._orders[order_id] = {
                "side": side,
                "price": price,
                "size": remaining,
                "seq": self._seq,
            }
        self._publish()
        if remaining and filled:
            return self._result(order_id, "PartiallyFilledResting", filled, remaining)
        if remaining:
            return self._result(order_id, "Accepted", filled, remaining)
        return self._result(order_id, "Filled" if filled else "NoOp", filled)

    def submit_market(
        self,
        order_id: int,
        side: StrictEngine.Side,
        qty: int,
    ) -> dict[str, int | str]:
        self._require_int(order_id, "order_id")
        self._require_int(qty, "qty")
        assert isinstance(side, self.Side)
        self._fills.clear()
        if qty <= 0 or order_id in self._orders:
            return self._result(order_id, "NoOp")
        filled, _remaining = self._match(order_id, side, qty, None)
        if filled:
            self._publish()
        return self._result(order_id, "Filled" if filled else "NoOp", filled)

    def cancel(self, order_id: int) -> dict[str, int | str]:
        self._require_int(order_id, "order_id")
        self._fills.clear()
        if order_id not in self._orders:
            return self._result(order_id, "NoOp")
        del self._orders[order_id]
        self._publish()
        return self._result(order_id, "Canceled")

    def modify(self, order_id: int, price: int, qty: int) -> dict[str, int | str]:
        self._require_int(order_id, "order_id")
        self._require_int(price, "price")
        self._require_int(qty, "qty")
        self.modify_calls.append((order_id, price, qty))
        self._fills.clear()
        order = self._orders.get(order_id)
        if order is None:
            return self._result(order_id, "NoOp")
        if price != order["price"] or qty > order["size"]:
            return self._result(order_id, "Rejected_BadQty")
        if qty == 0:
            del self._orders[order_id]
            self._publish()
            return self._result(order_id, "Canceled")
        order["size"] = qty
        self._publish()
        return self._result(order_id, "Accepted", resting=qty)


def test_stub_adapter_lookup_cancel_snapshot_and_reset_identity():
    raw = StubOrderBook()
    adapter = StubBookAdapter(raw)
    first = adapter.rest(Side.Bid, 100, 12, order_id=7)
    rejected = adapter.rest(Side.Bid, 99, 5, order_id=7)

    assert rejected.size == 0
    assert adapter.lookup(7) is first
    assert adapter.cancel_id(7, 5) == 5
    assert first.size == 7
    snap = adapter.snapshot()
    adapter.rest(Side.Bid, 99, 5, order_id=8)
    assert int(snap["bid_px"][0]) == 100
    assert int(snap["bid_sz"][0]) == 7

    adapter.reset()

    assert adapter.book is raw
    assert adapter.lookup(7) is None
    assert all(int(value) == 0 for value in raw.view()["bid_px"])
    assert raw.seq == raw.ts_ns == raw.cum_volume == raw.version == 0
    assert BOOK_STATE_DTYPE.itemsize == 40 * DEPTH + 48


def test_order_book_env_keeps_injected_stub_identity_across_resets():
    raw = StubOrderBook()
    env = OrderBookEnv(inventory=100, horizon=3, seed=11, child_max=40, book=raw)
    adapter = env.book
    assert isinstance(adapter, StubBookAdapter)

    env.reset(seed=11)
    first = adapter.snapshot()
    env.step(-1.0)
    env.reset(seed=11)
    second = adapter.snapshot()

    assert env.book is adapter
    assert adapter.book is raw
    for key in ("bid_px", "bid_sz", "bid_ct", "ask_px", "ask_sz", "ask_ct"):
        np.testing.assert_array_equal(first[key], second[key])


def test_engine_adapter_reconciles_crossed_makers_on_strict_surface():
    engine = StrictEngine()
    adapter = adapt(engine)
    assert isinstance(adapter, EngineAdapter)

    maker = adapter.rest(Side.Ask, 101, 100, order_id=41)
    crossed = adapter.rest(Side.Bid, 101, 40, order_id=42)

    assert crossed.size == 0
    assert engine.submit_limit_calls[-1][1] is StrictEngine.Side.Bid
    assert engine.submit_limit_calls[-1][-1] is StrictEngine.TimeInForce.GTC
    assert adapter.lookup(41) is maker
    assert maker.size == 60
    assert adapter.cancel_id(41, 20) == 20
    assert engine.modify_calls[-1] == (41, 101, 40)
    assert adapter.lookup(41) is maker
    assert maker.size == 40
    assert adapter.cancel_resting(maker) == 40
    assert adapter.lookup(41) is None


def test_engine_adapter_take_tracks_maker_residuals_on_strict_surface():
    adapter = EngineAdapter(StrictEngine())
    first = adapter.rest(Side.Ask, 103, 20, order_id=20)
    second = adapter.rest(Side.Ask, 105, 40, order_id=21)

    result = adapter.take(Side.Bid, 30)

    assert result.filled == 30
    assert result.notional_ticks == 20 * 103 + 10 * 105
    assert adapter.lookup(first.order_id) is None
    assert adapter.lookup(second.order_id) is second
    assert second.size == 30


def test_engine_adapter_forwards_view_snapshot_and_resets_via_factory():
    raw = StrictEngine()
    adapter = EngineAdapter(raw, engine_factory=StrictEngine)
    live = adapter.view()
    frozen = adapter.snapshot()

    adapter.rest(Side.Bid, 100, 10, order_id=1)

    assert int(live["bid_px"][0]) == 100
    assert int(live["bid_sz"][0]) == 10
    assert int(frozen["bid_px"][0]) == 0
    assert int(frozen["bid_sz"][0]) == 0

    adapter.reset()

    assert adapter.engine is not raw
    assert adapter.lookup(1) is None
    assert int(adapter.view()["bid_px"][0]) == 0
    assert int(raw.view()["bid_px"][0]) == 100


@pytest.fixture(params=["stub", "strict", "compiled"])
def parity_book(request):
    if request.param == "stub":
        return StubBookAdapter()
    if request.param == "strict":
        return EngineAdapter(StrictEngine())
    if ne is None:
        pytest.skip("nexus_engine is not built")

    def factory():
        return ne.Engine(1, 1000, 1024)

    return EngineAdapter(factory(), engine_factory=factory)


def test_adapter_duplicate_and_cancel_handle_parity(parity_book):
    book = parity_book
    maker = book.rest(Side.Ask, 101, 10, order_id=7)
    duplicate = book.rest(Side.Ask, 102, 20, order_id=7)
    assert duplicate.size == 0
    assert book.lookup(7) is maker
    assert maker.size == 10
    assert book.cancel_id(7, 0) == book.cancel_id(7, -1) == 0
    assert book.cancel_id(7, 3) == 3
    assert maker.size == 7
    assert book.cancel_resting(maker) == 7
    assert maker.size == 0
    assert book.lookup(7) is None
    assert book.cancel_resting(maker) == book.cancel_id(7) == 0

    second = book.rest(Side.Bid, 99, 4, order_id=8)
    assert book.cancel_id(8) == 4
    assert second.size == 0


def test_partial_cancel_modify_preserves_fifo_and_counts(parity_book):
    book = parity_book
    first = book.rest(Side.Ask, 101, 10, order_id=7)
    second = book.rest(Side.Ask, 101, 10, order_id=8)
    assert book.cancel_id(7, 4) == 4
    assert first.size == 6
    assert int(book.view()["ask_ct"][0]) == 2
    result = book.take(Side.Bid, 8)
    assert result.filled == 8
    assert result.notional_ticks == 808
    assert book.lookup(7) is None
    assert first.size == 0
    assert book.lookup(8) is second
    assert second.size == 8
    assert int(book.view()["ask_ct"][0]) == 1
    assert int(book.view()["ask_sz"][0]) == 8


def test_adapter_take_walks_beyond_abi_visible_depth(parity_book):
    book = parity_book
    prices = list(range(100, 100 + DEPTH + 3))
    for order_id, price in enumerate(prices, start=1):
        book.rest(Side.Ask, price, 2, order_id=order_id)
    assert len(book.view()["ask_px"]) == DEPTH
    result = book.take(Side.Bid, 2 * len(prices))
    assert result.filled == 2 * len(prices)
    assert result.notional_ticks == 2 * sum(prices)
    assert int(book.view()["ask_px"][0]) == 0
    assert all(book.lookup(order_id) is None for order_id in range(1, len(prices) + 1))


requires_compiled_engine = pytest.mark.skipif(
    ne is None,
    reason="C++ bridge not built yet — build bindings/ then re-run (see CONTRACT.md).",
)


@requires_compiled_engine
def test_adapt_keeps_adapter_identity_while_reset_replaces_opaque_engine():
    raw = ne.Engine()
    env = OrderBookEnv(inventory=200, horizon=4, seed=4, child_max=50, book=raw)
    adapter = env.book
    assert isinstance(adapter, EngineAdapter)
    assert adapter.engine is raw

    env.reset()

    assert env.book is adapter
    assert adapter.engine is not raw
    assert int(adapter.view()["bid_px"][0]) > 0
    assert int(adapter.view()["ask_px"][0]) > 0


@requires_compiled_engine
def test_engine_rest_crossing_reconciles_existing_handle():
    adapter = EngineAdapter(ne.Engine())
    maker = adapter.rest(Side.Ask, 10_050, 100, order_id=2)
    crossed = adapter.rest(Side.Bid, 10_050, 40, order_id=3)

    assert crossed.size == 0
    assert adapter.lookup(2) is maker
    assert maker.size == 60
    assert adapter.cancel_id(2, 20) == 20
    assert int(adapter.view()["ask_sz"][0]) == 40


@requires_compiled_engine
def test_engine_adapter_forwards_zero_copy_view_and_owning_snapshot():
    adapter = EngineAdapter(ne.Engine())
    live = adapter.view()
    frozen = adapter.snapshot()

    adapter.rest(Side.Bid, 10_000, 500, order_id=1)

    assert int(live["bid_px"][0]) == 10_000
    assert int(live["bid_sz"][0]) == 500
    assert int(frozen["bid_px"][0]) == 0
    assert int(frozen["bid_sz"][0]) == 0


@requires_compiled_engine
def test_engine_limit_market_cancel_fills():
    adapter = EngineAdapter(ne.Engine())
    adapter.rest(Side.Ask, 10_050, 100, order_id=2)
    result = adapter.take(Side.Bid, 40)
    assert result.filled == 40
    assert result.notional_ticks == 40 * 10_050
    adapter.rest(Side.Bid, 10_000, 500, order_id=1)
    assert adapter.cancel_id(1, 100) == 100
    assert int(adapter.view()["bid_sz"][0]) == 400
    assert adapter.cancel_id(1) == 400
    assert int(adapter.view()["bid_px"][0]) == 0


@requires_compiled_engine
def test_env_inventory_and_obs_on_engine():
    Q, T = 240, 6
    env = OrderBookEnv(inventory=Q, horizon=T, seed=9, child_max=50, book=ne.Engine())
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,)
    while True:
        obs, _r, term, trunc, info = env.step(-1.0)
        if term or trunc:
            break
    filled = sum(sz for _, _, sz in env.fills)
    assert filled + env.inventory == Q
    assert obs.shape == (44,)
    assert np.isfinite(info["shortfall_bps"])
    assert env.execution_vwap() > 0


@requires_compiled_engine
def test_twap_on_engine():
    env = OrderBookEnv(inventory=120, horizon=4, seed=5, child_max=40, book=ne.Engine())
    result = run_episode(env, "twap", seed=5)
    assert result.filled + result.leftover == 120
    assert result.steps > 0

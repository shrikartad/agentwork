

"""
Nexus-LOB :: cross-domain order-book state contract (Python side).
Owner: Person B (python_quant).

Mirrors the frozen C++ ``nexus::BookStateView``
(cpp_engine/include/nexus/book_state.hpp). Two things live here:

  * Constants + BOOK_STATE_DTYPE — the schema the env / agent code targets.
  * StubOrderBook — a pure-Python book emitting the *identical* view()/snapshot()
    interface as the C++ ``nexus_engine.Engine``, so OrderBookEnv can be built and
    trained before the C++ engine lands. Swapping ``StubOrderBook`` for
    ``nexus_engine.Engine`` requires no change to the observation code.

See bindings/CONTRACT.md for the specification.
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np

CONTRACT_VERSION = 1
DEPTH = 10  # keep in lock-step with nexus::kDepth

# Cross-boundary element dtypes (prices are INTEGER TICKS, never floats).
PX_DTYPE = np.int64
SZ_DTYPE = np.uint64
CT_DTYPE = np.uint32


class Side(IntEnum):
    Bid = 0
    Ask = 1
    NONE = 2


# Structured dtype used by the stub and by the shared-memory path (subsystem 5).
# Field order matches the C++ struct; align=True requests C-compatible padding.
# Byte-for-byte parity with the C++ struct (itemsize == 40*DEPTH + 48) is
# asserted by tests/test_abi_parity.py before the shmem path is trusted.
BOOK_STATE_DTYPE = np.dtype(
    {
        "names": [
            "seq", "ts_ns", "cum_volume", "last_trade_sz", "last_trade_px",
            "bid_px", "bid_sz", "ask_px", "ask_sz",
            "version", "bid_ct", "ask_ct",
            "last_trade_side",
        ],
        "formats": [
            np.uint64, np.int64, np.uint64, np.uint64, np.int64,
            (np.int64, DEPTH), (np.uint64, DEPTH), (np.int64, DEPTH), (np.uint64, DEPTH),
            np.uint32, (np.uint32, DEPTH), (np.uint32, DEPTH),
            np.uint8,
        ],
    },
    align=True,
)


def empty_state() -> np.ndarray:
    """A single zeroed BOOK_STATE_DTYPE record."""
    return np.zeros((), dtype=BOOK_STATE_DTYPE)


class StubOrderBook:
    """Minimal price-time book with the same Python interface as the C++ Engine.

    Not performance-tuned — its only jobs are (1) to unblock OrderBookEnv
    development and (2) to act as the reference oracle the C++ matching engine is
    diff-tested against. Prices are integer ticks throughout.
    """

    def __init__(self, depth: int = DEPTH) -> None:
        self.depth = depth
        self._bids: dict[int, list[int]] = {}  # price(ticks) -> [size, order_count]
        self._asks: dict[int, list[int]] = {}
        self.seq = 0
        self.ts_ns = 0
        self.cum_volume = 0
        self.last_trade_px = 0
        self.last_trade_sz = 0
        self.last_trade_side = int(Side.NONE)
        self.version = 0

    # --- mutations (resting liquidity) -----------------------------------
    def add(self, side: Side, price: int, size: int, orders: int = 1) -> None:
        book = self._bids if side == Side.Bid else self._asks
        lvl = book.setdefault(int(price), [0, 0])
        lvl[0] += int(size)
        lvl[1] += int(orders)
        self._advance()

    def cancel(self, side: Side, price: int, size: int, orders: int = 0) -> None:
        book = self._bids if side == Side.Bid else self._asks
        lvl = book.get(int(price))
        if lvl is None:
            return
        lvl[0] = max(0, lvl[0] - int(size))
        lvl[1] = max(0, lvl[1] - int(orders))
        if lvl[0] == 0:
            book.pop(int(price), None)
        self._advance()

    def record_trade(self, side: Side, price: int, size: int) -> None:
        self.last_trade_side = int(side)
        self.last_trade_px = int(price)
        self.last_trade_sz = int(size)
        self.cum_volume += int(size)
        self._advance()

    def reset(self) -> None:
        """Clear the book in place while preserving an injected object's identity."""
        self._bids.clear()
        self._asks.clear()
        self.seq = 0
        self.ts_ns = 0
        self.cum_volume = 0
        self.last_trade_px = 0
        self.last_trade_sz = 0
        self.last_trade_side = int(Side.NONE)
        self.version = 0

    def _advance(self) -> None:
        self.seq += 1
        self.ts_ns += 1
        self.version += 1

    # --- ladder extraction ------------------------------------------------
    def _ladder(self, side: Side) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        book = self._bids if side == Side.Bid else self._asks
        # bids: highest price first; asks: lowest price first.
        prices = sorted(book.keys(), reverse=(side == Side.Bid))[: self.depth]
        px = np.zeros(self.depth, dtype=PX_DTYPE)
        sz = np.zeros(self.depth, dtype=SZ_DTYPE)
        ct = np.zeros(self.depth, dtype=CT_DTYPE)
        for i, p in enumerate(prices):
            size, count = book[p]
            px[i], sz[i], ct[i] = p, size, count
        return px, sz, ct

    def _payload(self) -> dict:
        bid_px, bid_sz, bid_ct = self._ladder(Side.Bid)
        ask_px, ask_sz, ask_ct = self._ladder(Side.Ask)
        return {
            "bid_px": bid_px, "bid_sz": bid_sz, "bid_ct": bid_ct,
            "ask_px": ask_px, "ask_sz": ask_sz, "ask_ct": ask_ct,
            "seq": self.seq, "ts_ns": self.ts_ns,
            "cum_volume": self.cum_volume,
            "last_trade_px": self.last_trade_px,
            "last_trade_sz": self.last_trade_sz,
            "last_trade_side": Side(self.last_trade_side),
        }

    # --- public interface (mirrors nexus_engine.Engine) -------------------
    def view(self) -> dict:
        """Shape-identical to Engine.view(). No aliasing hazard in the stub."""
        return self._payload()

    def snapshot(self) -> dict:
        return self._payload()

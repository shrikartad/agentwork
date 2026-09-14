"""Gymnasium optimal-execution environment.

Liquidate a long inventory ``Q`` over a fixed horizon ``T`` against an
injectable L2 book (``StubBookAdapter`` today, ``EngineAdapter`` later).

Observation
-----------
``Box(float32, shape=(44,))`` layout:

=======  =====  ========================================================
index    name   definition
=======  =====  ========================================================
0..9     bidΔ   (mid − bid_px[i]) / OFFSET_SCALE; 1.0 if the level is empty
10..19   bidSz  bid_sz[i] / SIZE_SCALE
20..29   askΔ   (ask_px[i] − mid) / OFFSET_SCALE; 1.0 if empty
30..39   askSz  ask_sz[i] / SIZE_SCALE
40       inv    remaining inventory / Q
41       tLeft  remaining steps / T
42       pnl    mark-to-market ticks / (Q * 10), clipped to [−3, 3]
43       spread (ask0 − bid0) / OFFSET_SCALE; OFFSET_SCALE if no BBO
=======  =====  ========================================================

``OFFSET_SCALE = 20`` ticks, ``SIZE_SCALE = 800`` shares. Level *prices*
stay integer ticks inside the book; only the observation vector is float.

Action
------
``Box(low=-1, high=1, shape=(1,))``. Selling a long:

* ``a <= -0.92`` → market sell the child (hit displayed bids).
* else ``offset = round(a * MAX_OFFSET)`` with ``MAX_OFFSET = 12`` ticks
  and a sell limit at ``round(mid) + offset``.
* If that limit is at or through the best bid, treat it as a capped market.
* Otherwise rest on the ask at ``max(limit, best_bid + 1)``.

Examples: ``-1`` market; ``0`` post at mid; ``0.7`` rest 8 ticks through
the ask; ``-0.4`` rest 5 ticks below mid (often inside the spread / at bid).

Child size is ``clamp(ceil(inv / t_left), 20, child_max)``. One live child
per step; the previous residual is cancelled first.

Reward
------
Arrival mid (the mid at ``reset``) is the implementation-shortfall benchmark.

Per step, after the child and a burst of exogenous book flow::

    IS_ticks = filled * mid_before − cash_from_child     # sell: >0 if below mid
    IS_norm  = IS_ticks / (Q * mid_before * 1e-4)
    inv_pen  = λ_inv * (inv/Q)^2
    time_pen = λ_t   * (inv/Q) * (t / T)
    adv      = λ_adv * max(0, mid_before − mid_after) / OFFSET_SCALE
    reward   = −is_coef · IS_norm − inv_pen − time_pen − adv

``is_coef`` (default 1.0) scales the price-quality (IS) term independent of
the inventory-pacing penalties. Raising it (e.g. 4) trains an agent that
prioritises fill price over fill certainty — the variant measured on the
shortfall_bps / vs-VWAP headline.

``lambda_sched`` (default 0.0) adds a TWAP-schedule-tracking term
``λ_sched · ((inv/Q) − (T−t)/T)²``. Non-zero, it holds the agent to the
time-weighted liquidation path and leaves only fill-price to optimise —
the clean way to chase the “lower slippage than VWAP” headline rather than
reward-hacking completion risk.

On horizon with leftover inventory the residual is market-dumped and an
extra ``2.5 * leftover/Q`` is subtracted. Mark-to-market PnL
(``cash + inv*mid − Q*arrival_mid``) is an observation feature only.

Inventory identity: ``sum(fill sizes) + remaining == Q`` at every
terminated / truncated step (terminal dump included).
"""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover — tests install gymnasium
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]

from ..book_port import Resting, adapt
from ..book_state import DEPTH, Side, StubOrderBook
from ..replay import spread_ticks

OBS_DIM = 44
OFFSET_SCALE = 20.0
SIZE_SCALE = 800.0
MAX_OFFSET = 12
DEFAULT_Q = 2_000
DEFAULT_T = 40

OBS_LABELS: list[str] = (
    [f"bidΔ{i}" for i in range(DEPTH)]
    + [f"bidSz{i}" for i in range(DEPTH)]
    + [f"askΔ{i}" for i in range(DEPTH)]
    + [f"askSz{i}" for i in range(DEPTH)]
    + ["inv", "tLeft", "pnl", "spread"]
)

_Base = gym.Env if gym is not None else object  # type: ignore[misc]


class OrderBookEnv(_Base):  # type: ignore[misc]
    metadata: ClassVar[dict[str, Any]] = {"render_modes": ()}

    def __init__(
        self,
        *,
        inventory: int = DEFAULT_Q,
        horizon: int = DEFAULT_T,
        seed: int = 0x51ED,
        lambda_inv: float = 0.15,
        lambda_time: float = 0.35,
        lambda_adv: float = 0.08,
        is_coef: float = 1.0,
        lambda_sched: float = 0.0,
        child_max: int = 220,
        book: Any | None = None,
        # --- high-volatility regime (defaults = no regime = current behavior) ---
        regime_prob: float = 0.0,
        vol_decay: float = 0.0,
        gap_prob: float = 0.0,
        gap_min: int = 800,
        gap_max: int = 1800,
        gap_down_prob: float = 0.75,
        vol_take_prob: float = 0.55,
        vol_take_min: int = 60,
        vol_take_max: int = 220,
        vol_add_min: int = 15,
        vol_add_max: int = 70,
        vol_add_offset_min: int = 2,
        vol_add_offset_max: int = 12,
        vol_events_min: int = 3,
        vol_events_max: int = 8,
        vol_feature: bool = False,
        # --- risk↔env seam (default off: existing tests / RNG path unchanged) ---
        lambda_risk: float = 0.0,
        risk_paths: int = 256,
        risk_steps: int = 16,
        risk_sigma: float = 0.25,
        # --- execution realism (Phase 2; defaults off = byte-identical to today) ---
        fee_bps: float = 0.0,
        rebate_bps: float = 0.0,
        impact_coef: float = 0.0,
        impact_participation: float = 0.1,
        queue_model: str = "",  # "" = off; "uniform" = placeholder queue-depth model
    ) -> None:
        self.inventory0 = int(inventory)
        self.horizon = int(horizon)
        self.lambda_inv = float(lambda_inv)
        self.lambda_time = float(lambda_time)
        self.lambda_adv = float(lambda_adv)
        self.is_coef = float(is_coef)
        self.lambda_sched = float(lambda_sched)
        self.child_max = int(child_max)
        self._seed0 = int(seed)
        self._rng = np.random.default_rng(self._seed0)
        self.book = adapt(book if book is not None else StubOrderBook())
        # regime params
        self.regime_prob = float(regime_prob)
        self.vol_decay = float(vol_decay)
        self.gap_prob = float(gap_prob)
        self.gap_min = int(gap_min)
        self.gap_max = int(gap_max)
        self.gap_down_prob = float(gap_down_prob)
        self.vol_take_prob = float(vol_take_prob)
        self.vol_take_min = int(vol_take_min)
        self.vol_take_max = int(vol_take_max)
        self.vol_add_min = int(vol_add_min)
        self.vol_add_max = int(vol_add_max)
        self.vol_add_offset_min = int(vol_add_offset_min)
        self.vol_add_offset_max = int(vol_add_offset_max)
        self.vol_events_min = int(vol_events_min)
        self.vol_events_max = int(vol_events_max)
        self.vol_feature = bool(vol_feature)
        self.lambda_risk = float(lambda_risk)
        self.risk_paths = int(risk_paths)
        self.risk_steps = int(risk_steps)
        self.risk_sigma = float(risk_sigma)
        self.fee_bps = float(fee_bps)
        self.rebate_bps = float(rebate_bps)
        self.impact_coef = float(impact_coef)
        self.impact_participation = float(impact_participation)
        self.queue_model = str(queue_model)
        if self.queue_model not in ("", "uniform"):
            raise ValueError(
                f"queue_model must be '' or 'uniform', got {self.queue_model!r}"
            )
        self.last_cvar = 0.0
        self._volatile = False
        obs_dim = 45 if self.vol_feature else OBS_DIM
        if gym is not None:
            self.observation_space = spaces.Box(
                low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
        self.t = 0
        self.inventory = 0
        self.cash_ticks = 0
        self.arrival_mid = 0
        self.agent_rest: Resting | None = None
        self.fills: list[tuple[int, int, int]] = []  # (t, px, sz)
        self._mkt_qty = 0
        self._mkt_notional = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        del options
        if seed is not None:
            self._seed0 = int(seed)
        self._rng = np.random.default_rng(self._seed0)
        self.book.reset()
        self._seed_book()
        self.t = 0
        self.inventory = self.inventory0
        self.cash_ticks = 0
        self.fills = []
        self.agent_rest = None
        self._volatile = False
        self._mkt_qty = 0
        self._mkt_notional = 0
        self.arrival_mid = self._mid() or 15_000
        obs = self._observe()
        return obs, {"arrival_mid": self.arrival_mid}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        a = float(np.asarray(action).reshape(-1)[0])
        a = max(-1.0, min(1.0, a))
        snap = self.book.snapshot()
        mid0 = self._mid(snap) or self.arrival_mid
        self._cancel_agent()
        want = min(self.inventory, self._child_size())
        filled = 0
        notional = 0
        mode = "limit"
        action_ticks = 0

        if a <= -0.92 or want <= 0:
            mode = "market"
            action_ticks = -MAX_OFFSET
            if want > 0:
                r = self.book.take(Side.Ask, want)
                filled, notional = r.filled, r.notional_ticks
        else:
            action_ticks = round(a * MAX_OFFSET)
            px = round(mid0) + action_ticks
            bid = int(snap["bid_px"][0])
            if bid and px <= bid:
                mode = "market"
                r = self.book.take(Side.Ask, want, limit_px=px)
                filled, notional = r.filled, r.notional_ticks
            else:
                limit = max(px, (bid + 1) if bid else px)
                self.agent_rest = self.book.rest(Side.Ask, limit, want)

        self._exogenous_flow()

        if self.agent_rest is not None:
            live = None
            if hasattr(self.book, "lookup"):
                live = self.book.lookup(self.agent_rest.order_id)
            residual = live.size if live is not None else 0
            got = want - residual
            if self.queue_model and got > 0:
                # Queue-position degradation: only a fraction of the *displayed*
                # depth ahead of the child is actually reachable this step.
                # Default "" = today's optimistic fill (queue ignored) = byte-identical.
                reachable = max(0, got - int(want * self._queue_ahead_frac()))
                got = min(got, reachable)
            if got > 0:
                px = live.price if live is not None else self.agent_rest.price
                filled += got
                notional += got * px
            if live is None or live.size <= 0:
                self.agent_rest = None

        if filled > 0:
            self.inventory -= filled
            self.cash_ticks += notional
            self.fills.append((self.t, notional // filled, filled))

        self.t += 1
        mid1 = self._mid() or mid0
        inv_frac = self.inventory / self.inventory0
        t_frac = self.t / self.horizon
        is_ticks = filled * mid0 - notional if filled else 0
        if filled and (self.fee_bps or self.rebate_bps or self.impact_coef):
            from ..execution.cost_model import impact

            # Maker/taker split: a resting limit filled by flow is passive (gets
            # the maker rebate, pays no fee); a market / capped-market child is
            # active (pays the taker fee, no rebate). is_ticks is a COST
            # (positive = worse), so a fee ADDS to it and a rebate subtracts.
            maker = mode == "limit"
            fee = self.fee_bps if not maker else 0.0
            reb = self.rebate_bps if maker else 0.0
            part = self.impact_participation
            is_ticks = is_ticks + filled * mid0 * (fee - reb) / 1e4
            if self.impact_coef:
                sig = self.risk_sigma or 0.25
                is_ticks = is_ticks + filled * mid0 * impact(part, sigma=sig, coef=self.impact_coef) / 1e4
        is_norm = is_ticks / (self.inventory0 * max(1, mid0) * 1e-4)
        adv = max(0, mid0 - mid1) / OFFSET_SCALE if filled else 0.0
        sched_frac = max(0.0, self.horizon - self.t) / self.horizon
        sched_dev = (inv_frac - sched_frac) * (inv_frac - sched_frac)
        reward = (
            -self.is_coef * is_norm
            - self.lambda_inv * inv_frac * inv_frac
            - self.lambda_time * inv_frac * t_frac
            - self.lambda_adv * adv
            - self.lambda_sched * sched_dev
        )
        if self.lambda_risk > 0.0:
            from ..risk import inventory_risk_penalty

            pen, res = inventory_risk_penalty(
                inv_frac,
                sigma=self.risk_sigma,
                seed=self._seed0 + self.t,
                n_paths=self.risk_paths,
                steps=self.risk_steps,
                lambda_risk=self.lambda_risk,
                horizon_frac=max(1.0 / 252.0, 1.0 - t_frac),
            )
            reward -= pen
            self.last_cvar = res.cvar

        terminated = self.inventory <= 0
        truncated = (not terminated) and self.t >= self.horizon
        if truncated and self.inventory > 0:
            dump = self.book.take(Side.Ask, self.inventory)
            self.inventory -= dump.filled
            self.cash_ticks += dump.notional_ticks
            if dump.filled:
                self.fills.append((self.t, dump.avg_px, dump.filled))
            reward -= 2.5 * (self.inventory / self.inventory0)

        vwap = self.execution_vwap()
        shortfall_bps = (
            ((self.arrival_mid - vwap) / self.arrival_mid) * 1e4 if vwap else 0.0
        )
        info = {
            "filled": filled,
            "inventory": self.inventory,
            "t_left": max(0, self.horizon - self.t),
            "mid": mid1,
            "arrival_mid": self.arrival_mid,
            "pnl_ticks": self.mark_to_market(mid1),
            "shortfall_bps": shortfall_bps,
            "action_ticks": action_ticks,
            "mode": mode,
            "vwap": vwap,
            "market_vwap": self.market_vwap(),
            "cvar": self.last_cvar,
        }
        return self._observe(), float(reward), bool(terminated), bool(truncated), info

    def execution_vwap(self) -> float:
        qty = sum(sz for _, _, sz in self.fills)
        if not qty:
            return 0.0
        return sum(px * sz for _, px, sz in self.fills) / qty

    def mark_to_market(self, mid: float | None = None) -> float:
        m = float(mid if mid is not None else (self._mid() or self.arrival_mid))
        return self.cash_ticks + self.inventory * m - self.inventory0 * self.arrival_mid

    def _observe(self) -> np.ndarray:
        s = self.book.view()
        mid = self._mid(s) or self.arrival_mid
        dim = 45 if self.vol_feature else OBS_DIM
        out = np.zeros(dim, dtype=np.float32)
        for i in range(DEPTH):
            bp, ap = int(s["bid_px"][i]), int(s["ask_px"][i])
            out[i] = (mid - bp) / OFFSET_SCALE if bp else 1.0
            out[10 + i] = float(s["bid_sz"][i]) / SIZE_SCALE
            out[20 + i] = (ap - mid) / OFFSET_SCALE if ap else 1.0
            out[30 + i] = float(s["ask_sz"][i]) / SIZE_SCALE
        out[40] = self.inventory / self.inventory0
        out[41] = max(0, self.horizon - self.t) / self.horizon
        pnl = self.mark_to_market(mid)
        out[42] = float(np.clip(pnl / (self.inventory0 * 10.0), -3.0, 3.0))
        spr = spread_ticks(s)
        out[43] = (spr if spr is not None else OFFSET_SCALE) / OFFSET_SCALE
        if self.vol_feature:
            out[44] = 1.0 if self._volatile else 0.0
        return out

    def _mid(self, state=None) -> float | None:
        s = state if state is not None else self.book.view()
        bb, ba = int(s["bid_px"][0]), int(s["ask_px"][0])
        if not bb or not ba:
            return None
        return (bb + ba) / 2.0

    def _child_size(self) -> int:
        left = max(1, self.horizon - self.t)
        twap = int(np.ceil(self.inventory / left))
        return max(20, min(self.child_max, twap))

    def _cancel_agent(self) -> None:
        if self.agent_rest is not None:
            self.book.cancel_resting(self.agent_rest)
            self.agent_rest = None

    def _queue_ahead_frac(self) -> float:
        """Fraction of the queue deemed ahead of the child this step (0..1).

        Placeholder until the order-level tracker lands (Phase 3
        ``queue_dynamics.py``): a uniform draw per step, deterministic under a
        fixed seed. Only called when ``queue_model`` is non-empty, so the
        default RNG stream (used for book flow / fills) is unaffected.
        """
        return float(self._rng.random())

    def _record_market_trade(self, result: Any) -> None:
        """Accumulate the tape's own prints (exogenous flow) into market VWAP.

        The agent's own fills/executions are deliberately EXCLUDED — market VWAP
        is the benchmark the strategy is measured against, so it must be
        independent of the strategy's own participation (plan_2.md §5 leak 3).
        """
        filled = int(getattr(result, "filled", 0))
        notional = int(getattr(result, "notional_ticks", 0))
        if filled > 0:
            self._mkt_qty += filled
            self._mkt_notional += notional

    def market_vwap(self) -> float:
        """Volume-weighted average price of the tape's exogenous prints (ticks)."""
        if self._mkt_qty <= 0:
            return 0.0
        return self._mkt_notional / self._mkt_qty

    def _seed_book(self) -> None:
        mid = 15_000
        for i in range(12):
            bsz = int(120 + self._rng.integers(0, 380))
            asz = int(120 + self._rng.integers(0, 380))
            self.book.rest(Side.Bid, mid - 1 - i, bsz)
            self.book.rest(Side.Ask, mid + 1 + i, asz)

    def _exogenous_flow(self) -> None:
        # --- Markov regime transition ---
        if self.regime_prob > 0 or self.vol_decay > 0:
            if self._volatile:
                if self._rng.random() < self.vol_decay:
                    self._volatile = False
            else:
                if self._rng.random() < self.regime_prob:
                    self._volatile = True

        if not self._volatile:
            # === calm regime: original code path (deterministic when params = 0) ===
            n = int(3 + self._rng.integers(0, 5))
            for _ in range(n):
                s = self.book.view()
                mid = self._mid(s) or 15_000
                roll = float(self._rng.random())
                if roll < 0.28:
                    side = Side.Bid if self._rng.random() < 0.5 else Side.Ask
                    self._record_market_trade(self.book.take(side, int(15 + self._rng.integers(0, 70))))
                else:
                    side = Side.Bid if self._rng.random() < 0.5 else Side.Ask
                    off = int(1 + self._rng.integers(0, 8))
                    px = round(mid) - off if side == Side.Bid else round(mid) + off
                    self.book.rest(side, px, int(30 + self._rng.integers(0, 160)))
        else:
            # === volatile regime: larger takes, thinner/wider adds, gap events ===
            if self.gap_prob > 0 and self._rng.random() < self.gap_prob:
                gap_sz = int(self._rng.integers(self.gap_min, self.gap_max + 1))
                # take() walks the OPPOSITE book: an Ask taker hits bids
                # (price falls = gap down); a Bid taker lifts asks (price up).
                side = Side.Ask if self._rng.random() < self.gap_down_prob else Side.Bid
                self._record_market_trade(self.book.take(side, gap_sz))
                self._ensure_bbo()

            n = int(self._rng.integers(self.vol_events_min, self.vol_events_max + 1))
            for _ in range(n):
                s = self.book.view()
                mid = self._mid(s) or 15_000
                roll = float(self._rng.random())
                if roll < self.vol_take_prob:
                    side = Side.Bid if self._rng.random() < 0.5 else Side.Ask
                    self._record_market_trade(
                        self.book.take(
                            side,
                            int(self._rng.integers(self.vol_take_min, self.vol_take_max + 1)),
                        )
                    )
                else:
                    side = Side.Bid if self._rng.random() < 0.5 else Side.Ask
                    off = int(self._rng.integers(self.vol_add_offset_min, self.vol_add_offset_max + 1))
                    px = round(mid) - off if side == Side.Bid else round(mid) + off
                    self.book.rest(
                        side, px,
                        int(self._rng.integers(self.vol_add_min, self.vol_add_max + 1)),
                    )

    def _ensure_bbo(self) -> None:
        """Replenish book if a gap drained it (avoids _mid() returning None)."""
        s = self.book.view()
        bb, ba = int(s["bid_px"][0]), int(s["ask_px"][0])
        if not bb:
            mid = 15_000
            for i in range(4):
                self.book.rest(Side.Bid, mid - 1 - i, int(100 + self._rng.integers(0, 100)))
            if not ba:
                for i in range(4):
                    self.book.rest(Side.Ask, mid + 1 + i, int(100 + self._rng.integers(0, 100)))
        elif not ba:
            for i in range(4):
                self.book.rest(Side.Ask, bb + 1 + i, int(100 + self._rng.integers(0, 100)))

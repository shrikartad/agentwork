"""Flow accounting invariants for the seeded execution environment."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from nexus_quant.book_state import Side
from nexus_quant.envs.order_book_env import ExogenousArrival, OrderBookEnv
from nexus_quant.replay import check_integrity


def _seed_controlled_bbo(env: OrderBookEnv) -> None:
    """Replace the random opening ladder with a small deterministic BBO."""
    for handle in list(env.book._orders.values()):
        env.book.cancel_resting(handle)
    env.book.rest(Side.Bid, 14_999, 100)
    env.book.rest(Side.Ask, 15_001, 100)


def test_passive_maker_execution_is_fully_credited_and_excluded_from_market_vwap():
    env = OrderBookEnv(inventory=100, horizon=2, seed=1257, queue_model="uniform")
    env.reset(seed=1257)
    _seed_controlled_bbo(env)
    env._next_exogenous_arrivals = lambda: (ExogenousArrival("take", Side.Bid, 70),)
    env._queue_arrival_delay = lambda _n: 0

    _, _, terminated, truncated, info = env.step(0.0)

    assert not terminated and not truncated
    assert info["mode"] == "limit"
    assert info["filled"] == 50
    assert sum(size for _, _, size in env.fills) == 50
    assert env.inventory0 - env.inventory == 50
    assert env.cash_ticks == 50 * 15_000
    # The 70-share external take hits the 50-share child, then 20 anonymous shares.
    assert env._mkt_qty == 20
    assert env._mkt_notional == 20 * 15_001
    assert info["market_vwap"] == pytest.approx(15_001.0)
    assert env.agent_rest is None
    assert env.queue_delays == (0,)


def _gappy_env(seed: int) -> OrderBookEnv:
    return OrderBookEnv(
        inventory=1_000,
        horizon=3,
        child_max=220,
        seed=seed,
        queue_model="uniform",
        regime_prob=1.0,
        vol_decay=0.0,
        gap_prob=1.0,
        gap_min=100_000,
        gap_max=100_000,
        gap_down_prob=1.0,
        vol_events_min=2,
        vol_events_max=2,
    )


def test_extreme_policies_share_immutable_pre_generated_arrivals():
    kwargs = {
        "inventory": 150,
        "horizon": 3,
        "child_max": 75,
        "seed": 82,
        "queue_model": "uniform",
        "regime_prob": 1.0,
        "vol_decay": 0.0,
        "gap_prob": 1.0,
        "gap_min": 50,
        "gap_max": 50,
        "gap_down_prob": 1.0,
        "vol_events_min": 0,
        "vol_events_max": 0,
    }
    market = OrderBookEnv(**kwargs)
    passive = OrderBookEnv(**kwargs)
    market.reset(seed=82)
    passive.reset(seed=82)
    _seed_controlled_bbo(market)
    _seed_controlled_bbo(passive)

    market.step(-1.0)
    passive.step(1.0)
    # The market child plus gap drained its BBO; the passive child did not.
    assert int(market.book.view()["bid_px"][0]) != int(passive.book.view()["bid_px"][0])
    market.step(-1.0)
    passive.step(1.0)

    assert market.exogenous_arrivals == passive.exogenous_arrivals
    assert len(market.exogenous_arrivals) == 2
    assert all(any(arrival.gap for arrival in batch) for batch in market.exogenous_arrivals)
    assert market.queue_delays == (None, None)
    assert all(delay is not None for delay in passive.queue_delays)
    with pytest.raises(FrozenInstanceError):
        market.exogenous_arrivals[0][0].qty = 1  # type: ignore[misc]


def test_gap_replenishment_restores_a_two_sided_book():
    env = OrderBookEnv(
        inventory=1_000,
        horizon=3,
        seed=82,
        regime_prob=1.0,
        vol_decay=0.0,
        gap_prob=1.0,
        gap_min=100_000,
        gap_max=100_000,
        gap_down_prob=1.0,
        vol_events_min=0,
        vol_events_max=0,
    )
    env.reset(seed=82)
    env.step(-1.0)

    arrivals = env.exogenous_arrivals[0]
    assert arrivals == (ExogenousArrival("take", Side.Ask, 100_000, gap=True),)
    state = env.book.view()
    assert int(state["bid_px"][0]) > 0
    assert int(state["ask_px"][0]) > 0


def test_seeded_arrivals_and_queue_delays_are_repeatable_and_bounded():
    left = _gappy_env(41)
    right = _gappy_env(41)
    left.reset(seed=41)
    right.reset(seed=41)

    for _ in range(3):
        obs_left, reward_left, term_left, trunc_left, info_left = left.step(0.8)
        obs_right, reward_right, term_right, trunc_right, info_right = right.step(0.8)
        np.testing.assert_array_equal(obs_left, obs_right)
        assert reward_left == pytest.approx(reward_right)
        assert (term_left, trunc_left) == (term_right, trunc_right)
        assert info_left["market_vwap"] == pytest.approx(info_right["market_vwap"])

    assert left.exogenous_arrivals == right.exogenous_arrivals
    assert left.queue_delays == right.queue_delays
    for delay, arrivals in zip(left.queue_delays, left.exogenous_arrivals):
        assert delay is not None
        assert 0 <= delay <= len(arrivals)


def test_queue_delay_cleans_terminal_residual_order():
    env = OrderBookEnv(inventory=50, horizon=1, seed=13, queue_model="uniform")
    env.reset(seed=13)
    _seed_controlled_bbo(env)
    env._next_exogenous_arrivals = lambda: (ExogenousArrival("take", Side.Bid, 30),)
    env._queue_arrival_delay = lambda n: n
    cancelled: list[int] = []
    cancel_resting = env.book.cancel_resting

    def record_cancel(handle):
        cancelled.append(handle.order_id)
        return cancel_resting(handle)

    env.book.cancel_resting = record_cancel
    _, _, terminated, truncated, info = env.step(0.0)

    assert not terminated and truncated
    assert env.queue_delays == (1,)
    assert info["filled"] == 0
    assert env._mkt_qty == 30
    assert env.agent_rest is None
    assert len(cancelled) == 1
    assert env.book.lookup(cancelled[0]) is None
    assert env.inventory == 0
    assert sum(size for _, _, size in env.fills) == env.inventory0


@pytest.mark.parametrize("survivor,price", [(Side.Ask, 14_980), (Side.Bid, 15_020)])
def test_generated_liquidity_never_crosses_a_surviving_quote(survivor, price):
    env = OrderBookEnv()
    env.reset(seed=7)
    env.book.reset()
    env.book.rest(survivor, price, 100)
    side = Side.Bid if survivor == Side.Ask else Side.Ask
    env._apply_exogenous_arrivals((ExogenousArrival("add", side, 50, relative_offset=1),))
    assert not list(check_integrity(env.book.view()))


def test_multilevel_child_and_terminal_fills_preserve_exact_integer_notional():
    env = OrderBookEnv(inventory=40, horizon=1, child_max=20)
    env.reset(seed=8)
    env.book.reset()
    env.book.rest(Side.Bid, 14_999, 21)
    env.book.rest(Side.Bid, 14_997, 30)
    env.book.rest(Side.Ask, 15_001, 100)
    env._next_exogenous_arrivals = lambda: ()
    _, _, _, truncated, info = env.step(-1.0)
    assert truncated and env.inventory == 0
    expected = 21 * 14_999 + 19 * 14_997
    assert env.cash_ticks == expected
    assert sum(px * size for _, px, size in env.fills) == expected
    assert all(isinstance(px, int) for _, px, _ in env.fills)
    assert info["vwap"] == pytest.approx(expected / 40)

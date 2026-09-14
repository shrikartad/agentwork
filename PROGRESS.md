# Nexus-LOB — Progress Report

**Status date:** 2026-09-14 · **Branch:** `hoplite/kranioi-df83c5d6` in `shrikartad/thelema`.
This authorized working copy preserves upstream Nexus-LOB PR #19 at
`69ae5af8faeaaf6694630cddc2c44b6ce0096ca6` and adds the remaining Person-B implementation.
No upstream PR was opened or modified by this follow-up.

**Execution update:** the full tape campaign and fresh-cache E7 study are now
running. Eight original tape URLs return 404; the corrected cohort uses 15 actual
full-session sources (seven retained, eight explicitly substituted), totaling
97.13 GB compressed. Source selection and reproducible commands are documented
in `docs/results/multi_day/README.md`. Neither URL availability nor partial
downloads count as completed statistical research.

**Current verification:** Python 3.12.3 on Linux, **274 passed / 1 skipped** in
`python_quant/tests`; **8 passed** in `bindings/tests`; **5/5 CTest** tests passed.
The skip requires a full local ITCH tape. With engine import disabled, the same
Python suite has **262 passed / 13 skipped**, retaining the always-on stub/fake tests.
Compileall, CI-scope Ruff, and `git diff --check` pass. Protected C++/CUDA/bindings
sources and the 448-byte state ABI are unchanged.

The empirical VWAP, E7 family-bootstrap, resumable batch, and Python adapter work
is complete. The dated studies below are historical: neither the full multi-day
campaign nor the published fairness study was rerun with the new code. This work
makes **no new out-of-sample execution-superiority claim**. See `progress_b.md`
for exact interfaces, limitations, and reproduction commands.

> **2026-09-13 — Person B, Part 2 (plain language):** we downloaded a real NASDAQ order-by-order
> tape (2019-12-30, AAPL + QQQ — 268 M messages), ran it through our parser and book with zero
> decode errors, and confirmed the C++ engine and the Python oracle produce the identical ladder
> on real bytes. On that tape the top-of-book imbalance genuinely predicts the next few mid
> moves (rank IC 0.14→0.46 as the horizon grows, tight CIs), passive limit orders that do get
> filled are almost always run over by the price right after (96–99 %), and our fill-probability
> model is calibrated. We also re-ran the PPO-vs-VWAP comparison *fairly* (same information for
> everyone, fees and queue on, five seeds, unseen regimes): the agent is **not** better than a
> decent adaptive schedule except when liquidity dries up — so the old "+50 % lower slippage"
> is retired. Full report: `docs/RESEARCH.md`; how to reproduce: `python_quant/scripts/run_all.py`.
>
> TL;DR: The cross-language state contract is frozen, the C++ matching engine is
> built and passing its own tests (86/86, 0 allocs/op), the pybind bridge drives the real engine,
> and the **shared-memory ring + interactive dashboard** (subsystems 4 and 5) are built and operational.
> Person B has landed the ITCH 5.0 streaming parser, replay engine, Gymnasium `OrderBookEnv`,
> baselines, GRPO trainer, and the dynamic risk↔env inventory penalty.
> Current test counts are in the 2026-09-14 verification above. The Part 2 implementation
> is present, but full multi-day statistical validation, updated fairness reports,
> and Person A's CUDA/hardware measurements remain separate work.

---

## 1. What the project is (30 seconds)

Nexus-LOB is a hybrid **C++/Python** limit-order-book (LOB) trading & market-
microstructure platform built as a **finance-placement portfolio project** (targets:
JPMC Quant Research, Nomura Algo, Goldman Systematics). Two people, ~8 weeks.

Five subsystems (from CLAUDE.md §2):
1. **Ultra-low-latency C++ matching engine** (Person A) — Limit / Market / FOK / IOC /
   Cancel / Modify, zero-allocation, integer-tick prices.
2. **Microstructure sim + RL execution agent** (Person B) — Gymnasium env; PPO/GRPO vs
   TWAP/VWAP/Avellaneda–Stoikov baselines.
3. **GPU risk engine (CUDA)** — Monte-Carlo VaR/CVaR over 100k+ paths (CPU path + exact NumPy parity ✅; CUDA kernel authored, blocked on GPU hardware).
4. **Zero-copy pipeline + interactive dashboard** — shmem/file-ring IPC → live L2 depth, latency histogram, VaR/CVaR tiles, trade ticker (✅ complete; subsystem 4/5).
5. *(Subsystem 5 in comments — the shmem ring → dashboard.)*

Headline resume targets:
- C++ matching engine: **>500k orders/sec, sub-µs latency**, zero-alloc (0 allocs/op proven in bench).
- Real-tape microstructure signals (Part 2): L1 imbalance rank IC 0.14→0.46, calibrated fill model (1.03–1.09), passive fills adversely selected 96–99% (`docs/RESEARCH.md`).
- PPO execution vs best fair baseline: re-verified fairly in Part 2 Phase 3 (`docs/results/rl_fairness.md`) — old naive +50.4% headline retired; genuine edge concentrated under liquidity shocks (+0.4…+1.4 bps).
- CUDA Monte-Carlo VaR/CVaR: CPU reference + exact NumPy parity verified (3/3); GPU kernel awaiting CUDA hardware.

---

## 2. The architecture seam (why things are built in this order)

The two people work in **parallel**, so we froze the *one data structure* that crosses
the C++↔Python boundary before writing the engine — the **state contract**
`BookStateView` (C++) ↔ `BOOK_STATE_DTYPE` (NumPy). Both sides build against that frozen
shape:

- **Person B** can build/train the RL env against a pure-Python `StubOrderBook` that
  emits the exact same `view()`/`snapshot()` interface as the real engine — **today**, no
  C++ build needed.
- **Person A** drops the real `LimitOrderBook` behind the same seam. `StubOrderBook`
  then becomes the **reference oracle** the real engine is diff-tested against.

Rules that must never be broken (they keep the two halves compatible):
- **Prices are integer ticks** (`int64`), never floats. `0` in a price slot = empty level.
- **Fixed depth** `kDepth = DEPTH = 10` per side; index 0 = best level; bids descend,
  asks ascend; empty levels zero-padded.
- **Layout is ABI-frozen:** `sizeof(BookStateView) == 40*10 + 48 == 448` bytes.
  Changing it = ABI break (bump the version, update the Python mirror, re-run parity tests).

See `bindings/CONTRACT.md` for the full spec.

---

## 3. What is DONE (and verified)

### 3a. C++ matching engine — **built & self-tested** ✅
| Piece | File | Notes |
|---|---|---|
| Frozen state contract | `cpp_engine/include/nexus/book_state.hpp` | ABI v1, `sizeof == 448`, `alignof == 8` |
| Value types | `cpp_engine/include/nexus/types.hpp` | `OrderId/Price/Qty`, `Fill`, `Status`, `ExecResult` |
| Zero-alloc order pool | `cpp_engine/include/nexus/order_pool.hpp` | fixed slab + intrusive free-list |
| Matching engine | `cpp_engine/include/nexus/limit_order_book.hpp` | Limit/Market/FOK/IOC/Cancel/Modify, FIFO, O(1) level lookup & id map |
| **Engine tests** | `cpp_engine/tests/lob_test.cpp` | **86/86 checks pass** |
| ABI lock | `cpp_engine/tests/abi_check.cpp` | prints 448 / alignof 8 ✅ |

Verified 2026-08-30 with MSYS2 g++: `lob_test` → `86 checks, 0 failed / ALL PASS`;
`abi_check` → `sizeof = 448 (expected 448)`. Engine behavior covered: resting & L2
ladder order, full/partial crosses, price-time FIFO, multi-level sweeps, IOC / FOK /
market semantics, cancel, modify (priority-keeping reduce vs. priority-losing reprice),
and every reject path (bad qty/price, dup id, pool-full).

### 3b. Build system ✅
- `CMakeLists.txt` — engine lib (header-only today → static lib when `.cpp` land), the
  `nexus_engine` pybind module, `abi_check` + `lob_test` under CTest, and CUDA hooks
  (on-but-stubbed).
- `pyproject.toml` — scikit-build-core packaging; pytest `pythonpath` includes
  `python_quant` and `bindings` so tests import both `nexus_quant` and the compiled
  module without a pip install.

### 3c. Python side — **authored, not yet run** ⚠️
- `python_quant/nexus_quant/book_state.py` — NumPy dtype mirror + `StubOrderBook`
  (the oracle).
- `python_quant/tests/test_contract_smoke.py` — pure-NumPy smoke test.
- `python_quant/nexus_quant/__init__.py` — package exports.

### 3d. Pybind bridge — **wired to the real engine, not yet compiled** ⚠️
`bindings/pybind_wrapper.cpp` no longer a placeholder. `Engine` now owns a real
`LimitOrderBook` and exposes order entry, fills, and the two view flavors (see §5).
`bindings/tests/test_abi_parity.py` updated to drive the real engine.

> **⚠️ Important status nuance:** the C++ engine is verified here. The **compiled
> `nexus_engine` module and the Python tests are NOT built/run yet** — this machine's
> shell has no real Python / CMake / pybind (see §7). They must be built in **WSL**.

### 3e. Shared-memory ring + flow (subsystem 5, C++) — **built & demoed** ✅
| Piece | File | Notes |
|---|---|---|
| Shared-memory SPSC ring | `cpp_engine/include/nexus/shm_ring.hpp` | OS shmem (POSIX + Windows), lock-free SPSC, drop-new-on-full, 448-B slots |
| Synthetic flow generator | `cpp_engine/include/nexus/flow_gen.hpp` | seeded LCG, mid random-walk + passive/aggressive mix |
| Ring tests | `cpp_engine/tests/ring_test.cpp` | **30,011 checks pass** (order + integrity; drop semantics) |
| Publisher + probe demos | `cpp_engine/demos/ring_producer.cpp`, `ring_probe.cpp` | live cross-process book demo — **verified 200 frames, 0 dropped** |

This is the transport the future dashboard consumes: the engine (real or synthetic flow)
publishes its `BookStateView` into the ring after every order; a reader process follows
the live book. Same frozen 448-byte payload end to end.

### 3f. Person B — ITCH replay + execution env (**authored, not yet run** ⚠️)
Merged in PR #2 (`feature/env-and-itch`). Runs today against `StubOrderBook` (no C++ build
needed); swaps to the real engine via `book_port.adapt(...)`.

| Piece | File | Notes |
|---|---|---|
| ITCH 5.0 parser | `python_quant/nexus_quant/itch_parser.py` | streaming, framed+raw, Add/MPID/Exec/ExecPx/Cancel/Delete/Replace/Trade |
| ITCH→L2 replay | `python_quant/nexus_quant/replay.py` | `ReplayEngine` + `check_integrity` (crossed/locked/unsorted/neg-size) |
| Injectable adapter | `python_quant/nexus_quant/book_port.py` | `StubBookAdapter` + `EngineAdapter`; **`cancel_id`/`lookup` added 2026-09-04** so replay works against the real engine |
| Gymnasium env | `python_quant/nexus_quant/envs/order_book_env.py` | 44-dim obs, IS reward + inv/time/adv penalities, terminal dump |
| Baselines | `python_quant/nexus_quant/baselines.py` | TWAP / VWAP / POV / Passive |
| Tests | `python_quant/tests/test_{itch_parser,order_book_env,replay}.py` | ✅ **PASSING** (2026-09-04) |
| Diff-test harness | `bindings/tests/test_diff_engine_stub.py` | ✅ **PASSING** — Engine-vs-Stub L2-ladder parity |

> ✅ **All of §3f is now RUN and PASSING** (2026-09-04, Windows/MSVC). `pytest
> python_quant/tests bindings/tests -v` → **34 passed**, incl. `test_abi_parity.py` (6) and
> `test_diff_engine_stub.py` (2) — the real engine and the stub oracle agree.

---

### 3g. Person A — Monte-Carlo VaR/CVaR risk engine (subsystem 3) ✅ (CPU; GPU blocked)
Authored on `feature/risk-engine` (2026-09-06). Key idea: the per-path RNG is a
**pure function of (seed, path, step)** via counter-based splitmix64, so CPU,
CUDA, and a NumPy oracle all draw *identical* paths → **bit-for-bit** parity
(not Monte-Carlo tolerance).

| Piece | File | State |
|---|---|---|
| Model + RNG (GBM / Merton jump-diffusion) | `cuda_risk/risk_common.hpp` | ✅ |
| CPU serial reference | `cuda_risk/risk_cpu.hpp` | ✅ (200k×252 ≈ 1.0 s here) |
| CUDA kernel + launcher (1 thread/path) | `cuda_risk/risk_cuda.{cu,h}` | ⚠️ authored, needs toolkit |
| CPU-vs-GPU bench + parity | `cuda_risk/risk_bench.cpp` | ✅ CPU; GPU on a CUDA box |
| pybind `compute_var_cvar` (CPU) | `bindings/pybind_wrapper.cpp` | ✅ |
| NumPy exact-parity oracle | `python_quant/tests/test_risk_parity.py` | ✅ 3/3 |
| C++ statistical tests | `cpp_engine/tests/risk_test.cpp` | ✅ CTest 5/5 |
| CMake wiring | `CMakeLists.txt` | ✅ `nexus_risk`/`risk_bench` under toolkit |

**Verified here (no GPU): pytest 49 passed, CTest 5/5.** The CUDA kernel and
~40× speedup cannot be compiled/measured on this machine — that's the blocker.

## 4. What is NOT done yet

- ✅ ~~Pybind module built + parity tests green~~ — **DONE 2026-09-04** (Windows/MSVC).
- ✅ ~~Run the authored Python (parser/replay/env/baselines/diff-test)~~ — **DONE 2026-09-04**.
- ✅ ~~RL execution agent (PPO) vs the baselines~~ — **DONE 2026-09-05** (beats all baselines on the
  env's reward; ≈ VWAP on shortfall — see `python_quant/nexus_quant/agents/README.md` for the
  honest numbers and the high-vol path to the slippage headline).
- ⚠️ **High-volatility headline (+50.4%) → RE-VERIFIED & RETIRED (2026-09-13)**: The naive
  +50.4% shortfall improvement vs a simple VWAP heuristic on synthetic Markov regimes was audited
  in Part 2 Phase 3 (`docs/results/rl_fairness.md`). Under symmetric information and against
  `adaptive_pov`, PPO shows no significant edge on high-vol/trending regimes (|Δ| ≲ 0.3 bps) and
  only shows an edge in liquidity shocks (+0.4…+1.4 bps). The naive claim is officially retired.
- ✅ **Person B polish & Part 2 quantitative research layer** — **ALL PHASES COMPLETE (2026-09-13)**:
  - GRPO policy optimization agent (`python_quant/nexus_quant/agents/grpo.py`).
  - Risk ↔ environment integration: CVaR inventory penalty in `OrderBookEnv` (`lambda_risk`).
  - Interactive Python order-book dashboard (`python_quant/scripts/serve_dashboard.py`) reading live shmem/file rings.
  - Full Part 2 Phases 0–5: real ITCH streaming, order-level queue dynamics, Kaplan–Meier fill models, adverse selection accounting, full-day E1–E6 real-tape results on AAPL/QQQ (`docs/RESEARCH.md`), and fair RL re-verification (`docs/results/rl_fairness.md`).
  - Landed on `feature/person-b-part2` (PR #19 on `Lokeshrao69/Nexus_LOB`).
- ⚠️ **CUDA VaR/CVaR risk engine (subsystem 3)** — CPU reference + exact NumPy parity + CTest
  **DONE & green** (2026-09-06, branch `feature/risk-engine`); the **GPU kernel + ~40× speedup
  are BLOCKED** on CUDA hardware (needs `nvcc`/toolkit).
- ✅ **Interactive order-book dashboard (subsystems 4 & 5)** — C++ ring producer + Python consumer + HTML ladder
  **DONE & operational**.

---

## 5. For Person B — the Python API you code against

Once `nexus_engine` is built, the seam surface is exactly the same shape as
`StubOrderBook`, so your env code can target either. The real engine adds order entry:

```python
import nexus_engine as ne

e = ne.Engine()                      # default price band 1..100_000 ticks
# e = ne.Engine(min_price=1, max_price=500_000, pool_capacity=1 << 18)

# Rest a GTC limit: place a bid at price 10_000 for 500 shares.
r = e.submit_limit(1, ne.Side.Bid, 10_000, 500, ne.TimeInForce.GTC)
r  # -> {"id": 1, "status": ne.Status.Accepted, "filled": 0, "resting": 500}

# Aggressive buy lifting the best ask(es) — fills come back per call.
e.submit_limit(10, ne.Side.Ask, 10_050, 100, ne.TimeInForce.GTC)
r2 = e.submit_limit(11, ne.Side.Bid, 10_050, 150, ne.TimeInForce.GTC)
r2  # -> {"id": 11, "status": ne.Status.Filled, "filled": 100, "resting": 0}
e.fills()  # -> [(10, 11, 10_050, 100, ne.Side.Bid)]  # (maker, taker, px, qty, aggressor)

# Quote introspection
e.best_bid(), e.best_ask(), e.spread(), e.live_orders()

# Observation (contract arrays): view() is ZERO-COPY (aliases engine memory —
# normalize/copy it now); snapshot() is a safe owning copy.
obs = e.view()          # {"bid_px","bid_sz","bid_ct","ask_px","ask_sz","ask_ct", ...}
snap = e.snapshot()
```

Key enum values:
- `Side`: `Bid`, `Ask`, `None_` (Python `None` is a keyword, hence `None_`).
- `TimeInForce`: `GTC` (rest residual), `IOC` (fill-then-kill), `FOK` (all-or-nothing).
- `Status`: `Accepted`, `Filled`, `PartiallyFilledResting`, `Canceled`,
  `Rejected_DupId`, `Rejected_BadPrice`, `Rejected_BadQty`, `Rejected_PoolFull`,
  `Rejected_FOK`, `NoOp`.

---

## 6. How to see everything work

```bash
# C++ only (works on Windows + MSYS2 g++, no build system needed):
g++ -std=c++20 -O2 -Wall -Wextra -I cpp_engine/include cpp_engine/tests/abi_check.cpp -o abi_check.exe && ./abi_check.exe
g++ -std=c++20 -O2 -Wall -Wextra -I cpp_engine/include cpp_engine/tests/lob_test.cpp -o lob_test.exe && ./lob_test.exe
g++ -std=c++20 -O2 -Wall -Wextra -I cpp_engine/include cpp_engine/tests/ring_test.cpp -o ring_test.exe && ./ring_test.exe

# Live shared-memory demo (subsystem 5) — run the producer in one terminal, the probe in another:
g++ -std=c++20 -O2 -I cpp_engine/include cpp_engine/demos/ring_producer.cpp -o ring_producer
g++ -std=c++20 -O2 -I cpp_engine/include cpp_engine/demos/ring_probe.cpp -o ring_probe
./ring_producer nex_aapl 4000 16384 0xC0FFEE 1     # terminal 1: book -> ring
./ring_probe nex_aapl 4000 5                          # terminal 2: watch it live

# Python dashboard (subsystems 4 & 5):
PYTHONPATH=python_quant python python_quant/scripts/serve_dashboard.py --synthetic --port 8080

# Run full Part 2 research suite:
python python_quant/scripts/run_all.py --quick

# Full build + Python tests (WSL / Ubuntu + real Python required):
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNEXUS_BUILD_PYBIND=ON
cmake --build build -j && ctest --test-dir build --output-on-failure
pytest bindings/tests/test_abi_parity.py -v      # contract parity + engine seam
pytest python_quant/tests/test_contract_smoke.py -v
```

---

## 7. Environment notes (why some things say "not run here")

This session runs on **Windows 11 + Git Bash / MSYS2**, repo on a **OneDrive** path.
Updated **2026-09-04**: `g++` (C++20 ✅), **real Python 3.12 installed** (Tier 1 pure-Python
tests **PASS**), and `cmake` via `pip`. The only remaining gap for the full build is
**MSVC Build Tools** (install in progress) — needed to compile the pybind `nexus_engine`
module for Tier 2 (parity + diff-test). CUDA still needs Linux or a Windows CUDA toolkit.

- ✅ C++-only compile/run checks work here (engine tests above).
- ✅ Pure-Python tests work here (`python -m pytest python_quant/tests -v` — 152 tests).
- ⏳ Tier 2 (compile `nexus_engine`) needs MSVC → then `cmake -S . -B build
  -DNEXUS_BUILD_PYBIND=ON && cmake --build build -j` + parity + diff-test.
- ⚠️ Keep `build/`, `data/`, venvs **out of the OneDrive-synced tree** — sync + build
  artifacts is a known breakage source (copy the repo off OneDrive if the build is slow/flaky).

See `CLAUDE.md` §8 for the full table and the exact Windows build steps.

---

## 8. Status and remaining work

1. ✅ **Tier 2 on Windows** — **PASSED 2026-09-04** (MSVC build + parity + diff-test).
2. ✅ **Person B: ITCH parser + `OrderBookEnv` + baselines** — merged.
3. ✅ **Diff-test harness** — done + passing.
4. ✅ **PPO/GRPO agent vs baselines** — done.
5. ✅ **Risk ↔ Environment integration** — done (`risk.py`, `lambda_risk` in `OrderBookEnv`).
6. ✅ **Interactive order-book dashboard** — done (`serve_dashboard.py`, shmem/file-ring decoding).
7. ✅ **Part 2 Phases 0–5 quant research layer** — done (`docs/RESEARCH.md`, `run_all.py`, PR #19).
8. ✅ **Dashboard Sanitization (Audit Priority 1)** — completed; retired +50.4% exploratory run labeled `Historical exploratory result — retired`, active fair study benchmark featured.
9. ✅ **E7 confidence intervals** — `evaluate_regime_ci()` and `paired_difference_ci()` support
   `fill_rate` and `mdd_ticks` alongside shortfall and market-VWAP slippage. Whole seed families
   are resampled without cutting dependent episode blocks; at least two equally sized families
   are required. Paired rows match by family and episode seed; higher fill rate and lower drawdown
   count as improvements. Intervals contain their estimate, initial execution loss enters MDD,
   and undefined relative percentages are `None` / `n/a`, not NaN or a superiority claim.
10. ✅ **Empirical VWAP conditioning** — an explicit `VolumeProfile` overrides
    `env.volume_profile`; its next-step forecast volume, normalized against uniform participation,
    sets VWAP aggression. Profiles are estimates from prior sessions, never future tape prints.
    The environment still controls child size. Without either profile the legacy VWAP arithmetic
    is unchanged; `schedule_twap` retains its existing cumulative-profile support.
11. 🟡 **Multi-day implementation ready; full statistical campaign not run** — all 15 dates
    in `PUBLIC_SAMPLE_DAYS` are supported by `scripts/batch_research_itch.py`. The runner uses
    sequential staged downloads, validated source/slice hashes, resumable E1–E6 outputs tied to
    a source-code fingerprint, and descriptive cross-day tables. Tests are network-free. A live
    transport/resume smoke fetched exactly 1 MiB each for `12302019` and `01302020`; both contained
    zero regular-session rows and are explicitly partial, not research evidence. Full days are
    roughly 3.5 GB compressed each; completing all 15 remains bandwidth-dependent (historical
    local throughput was about 300 KB/s). No full-day data or generated tape slices were committed.
12. ⏳ **Remaining Project Work (Person A):**
    - Verify GPU risk engine on a CUDA machine (`nexus_risk` + `risk_bench` with `nvcc`).
    - Hardware benchmarks for zero-copy shmem ring throughput.

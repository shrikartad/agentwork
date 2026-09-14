# Full-session ITCH campaign

## Source selection, 2026-09-14

The earlier 15-date catalogue included eight dates whose full gzip files now
return HTTP 404. Nasdaq's directory retains their `.gz.md5sum` files, not their
tapes. They are **unavailable**, not completed research days.

The execution cohort is the earliest 15 available, explicitly dated full-session
files from the same [official Nasdaq directory](https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/).
It retains seven original dates and substitutes 2019-10-18, 2021-07-13,
2021-08-13, and 2025-12-08 through 2025-12-12. The 2025-11-28 half-session is
excluded because the study uses a fixed 09:30–16:00 regular session. Selection
was based on source availability before examining any new research outcomes.

[`source_availability.json`](source_availability.json) records the selected
files, exact source URLs, HTTP status, sizes, modification dates, unavailable
original dates, and replacement-file framing checks. The cohort is
**97,129,873,038 compressed bytes**, not the earlier rough 3.5 GB/day estimate.
These are convenience samples across different years, not a random sample of
market days. No pooled event-level cross-day confidence interval is justified.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python python_quant/scripts/batch_research_itch.py \
  --days all --symbols AAPL,QQQ --download-workers 4 --skip-existing
```

There is **no compressed-byte cap and no event cap**. Each source is processed
through gzip EOF, retaining only the requested symbol slices in ignored `data/`.
Two to four download workers enable identity-pinned HTTP Range requests and a
SHA-256-checked compressed-part cache outside the batch staging directory.
Completed parts survive an interruption; cache cleanup follows successful gzip
validation and symbol-manifest creation. One worker retains the legacy streaming
path. Byte-capped smoke requests remain single-stream and explicitly partial.

Analysis uses the existing E1–E6 protocol: chronological 60/20/20 splits,
25-event split gaps, horizons 1/5/10/25, and 200 within-day moving-block
bootstrap replications. Queue/fill and adverse-selection studies are unchanged.
Per-day results carry source/slice hashes, analysis-code fingerprints, session
event timestamps, and uncapped-coverage metadata. Only derived reports and
metadata are committed; licensed source bytes and policy files are not.

## Execution status

**1 of 15 sessions is verified complete: 2019-12-30.** Its uncapped 3,524,013,057-byte
gzip produced 1,484,259 AAPL and 2,209,131 QQQ regular-session rows. Both symbols
have zero truncated messages, integrity issues, or unknown order IDs.

The completed analysis was handed over from commit `9ddf4d8`; the takeover
independently downloaded the entire source and verified identical gzip and symbol
slice hashes. Original execution/source fingerprints are preserved in
[`12302019/execution_manifest.json`](12302019/execution_manifest.json), with the
independent checks in [`12302019/takeover_validation.json`](12302019/takeover_validation.json).
These results were executed against `4b10862`, not silently relabeled as a later
checkout. The simulator corrections for E7 are a separate study.

| Symbol | L1 imbalance rank IC, h=1 [95% CI] | h=25 [95% CI] |
|---|---|---|
| AAPL | 0.1395 [0.1357, 0.1436] | 0.2289 [0.2163, 0.2430] |
| QQQ | 0.1549 [0.1522, 0.1578] | 0.4624 [0.4542, 0.4734] |

The other 14 sessions remain incomplete; the next full source download is
2019-01-30. Availability and partial downloads are not research results. A day
counts only after its research manifest and both symbol reports validate.
The generated cross-day index is descriptive, not pooled statistical evidence.

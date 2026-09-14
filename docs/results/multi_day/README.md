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

The full campaign is in progress. Availability and prefix-format checks are not
research results. A day is complete only when its validated research manifest
and both symbol reports exist; the final execution review will record actual
coverage, integrity, results, and remaining limitations.

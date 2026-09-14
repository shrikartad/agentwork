# Multi-day NASDAQ ITCH research batch summary

> This is a descriptive index of separate, time-ordered E1–E6 day studies. It does **not** pool event rows, fit a cross-day model, or make a cross-day performance claim. Partial GZIP prefixes are explicitly not full-day evidence.

## Batch status and coverage

| day | batch status | analysis status | coverage | symbols | regular-session rows |
|---|---|---|---|---|---|
| 12302019 | completed | completed | full_day | AAPL, QQQ | AAPL: 1,484,259, QQQ: 2,209,131 |

## E1 descriptive test-split index — L1 imbalance

Each IC and confidence interval remains a per-day estimate. Rows marked `partial_gzip_prefix` are shown for traceability, not pooled inference.

| day | symbol | coverage | h (events) | n test | rank IC test | 95% CI |
|---|---|---|---:|---:|---:|---|
| 12302019 | AAPL | full_day | 1 | 296827 | 0.1395 | [0.1357, 0.1436] |
| 12302019 | AAPL | full_day | 5 | 296827 | 0.2201 | [0.2131, 0.2285] |
| 12302019 | AAPL | full_day | 10 | 296826 | 0.2381 | [0.2294, 0.2488] |
| 12302019 | AAPL | full_day | 25 | 296823 | 0.2289 | [0.2163, 0.2430] |
| 12302019 | QQQ | full_day | 1 | 441802 | 0.1549 | [0.1522, 0.1578] |
| 12302019 | QQQ | full_day | 5 | 441800 | 0.2961 | [0.2905, 0.3010] |
| 12302019 | QQQ | full_day | 10 | 441799 | 0.3751 | [0.3678, 0.3819] |
| 12302019 | QQQ | full_day | 25 | 441796 | 0.4624 | [0.4542, 0.4734] |

## E5/E6 availability index

These fields expose what each separate tape can support; they are not aggregated estimates.

| day | symbol | coverage | regular orders | passive fills | KM P(fill by 50 events) | h=5 adverse drift | n |
|---|---|---|---:|---:|---:|---:|---:|
| 12302019 | AAPL | full_day | 791477 | 46357 | 0.0335 | -50.04 | 46357 |
| 12302019 | QQQ | full_day | 1141142 | 17662 | 0.0120 | -25.72 | 17662 |

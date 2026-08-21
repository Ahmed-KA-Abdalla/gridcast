# gridcast

A capture and evaluation harness for the Great Britain carbon intensity forecast
published by the National Energy System Operator (NESO).

The published forecast is revised every half hour and no history of it is
retained: the API serves the current estimate for a period and overwrites it as
that estimate changes. Forecast accuracy at a given lead time therefore cannot
be measured after the fact from the API alone. This repository records each
forecast at the moment it is issued and records the realised value when it
settles, so that the two can later be scored against each other.

The intended addition is a learned correction to the published forecast,
promoted into use only when it beats that forecast out of sample. None of the
modelling exists yet; the current state is below.

## Status

Built: the API client, the parsers, schema validation, raw snapshot storage, the
command-line interface, the scheduled capture workflow, a daily contract check
against the live API, and the loader that joins issued forecasts to the outcomes
they were forecasting.

Not built: feature construction, the baseline and candidate models, the
promotion gate, drift monitoring, and the published evaluation report.

## Data

Source: the NESO Carbon Intensity API, `https://api.carbonintensity.org.uk`,
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). No API key
is required. The API definition is at
<https://carbon-intensity.github.io/api-definitions/>.

Three endpoints are captured every half hour:

| Endpoint | Purpose |
| --- | --- |
| `/intensity` | The period in progress |
| `/intensity/{from}/fw48h` | The forecast as issued, out to 48 hours |
| `/intensity/{from}/pt24h` | Realised values for the past 24 hours |
| `/generation` | Fuel shares for the period in progress |

Raw responses are written verbatim under `data/raw/YYYY/MM/DD/`, each wrapped in
an envelope recording the endpoint requested and the time the response arrived.
Parsed derivatives are not stored: the parser will change, and a stored payload
can be reprocessed under a corrected parser where a stored derivative cannot.

Partitioning is by the date of capture, not the date the data describe. A
backfill run therefore writes every window it fetched under the day it ran; the
range each file covers is recorded in the envelope and in the filename.

The raw record is committed to the repository rather than kept in object
storage. It is small, its accumulation over time is the point of the project,
and holding it in git means the whole thing can be reproduced by cloning.

## Conventions worth knowing

**Capture time is part of the data.** Every parsed row carries `captured_at` and
a `horizon_hours` lead time. Without them a forecast is indistinguishable from a
later revision of itself, which makes accuracy by lead time unmeasurable and
makes any model trained on revised forecasts subject to leakage.

**Outcomes are harvested with redundancy.** Each run captures the past 24 hours
rather than only the period in progress. A settled value is published some
minutes after its period ends, so a run asking only about the present loses any
period whose run was delayed or dropped, and loses it permanently. Capturing the
past day means a period is reported by 48 successive runs.

**Only forecasts captured at issue time count as forecasts.** The `forecast`
field of a historical range response is a revised value, produced with
information unavailable at the lead time it appears to belong to. The loader
builds its forecast record exclusively from `forecast_fw48h` snapshots and
tests that the exclusion holds. Using the revised field would leak the outcome
into the predictor without disturbing any metric.

**Half-hour positions are UTC, not settlement periods.** GB settlement periods
are numbered against the local clock day, which has 46 or 50 of them at the
daylight-saving transitions. Rows carry `utc_half_hour` for the position on the
UTC grid and local clock time separately. The two agree for most of the year and
disagree exactly on the days when the system behaves unusually.

**Filenames come from a validated label, not the request path.** Request paths
contain colons. On NTFS a colon separates a filename from an alternate data
stream, so writing to such a path succeeds, reports success, and produces a file
no directory listing shows. Labels are validated rather than sanitised, so an
unexpected value fails at the call site.

**The live API does not match its published schema.** `/generation` returns a
bare object where the specification promises an array. Both parsers normalise
either form. Fixtures recorded from a specification cannot detect that the
specification and the service disagree, which is what the daily contract check
is for.

## Use

```
pip install -e ".[dev]"

gridcast snapshot                                        # one capture
gridcast backfill --start 2024-01-01 --end 2026-01-01    # historical actuals
gridcast report                                          # score what is stored
```

Backfill splits its range into fourteen-day windows, the maximum the range
endpoint accepts. It recovers realised values only. Forecasts as issued cannot
be recovered, for the reason given at the top.

## Tests

```
pytest -m "not network"     # offline, against recorded fixtures
pytest -m network           # exercises the live API
```

The offline suite passes without network access. The network-marked tests check
that the live API still returns the shape the parsers assume; they run daily
rather than on every commit.

## Prior art

[nmpowell/carbon-intensity-forecast-tracking](https://github.com/nmpowell/carbon-intensity-forecast-tracking)
scrapes the same API on a schedule to compare forecasts against realised values.
This repository starts from the same measurement problem; the intended addition
is the learned correction and promotion gate described above.

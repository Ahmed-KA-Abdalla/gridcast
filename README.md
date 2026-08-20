# gridcast

A capture and evaluation harness for the Great Britain carbon intensity forecast
published by the National Energy System Operator (NESO).

The published forecast is revised every half hour and no history of it is
retained: the API serves the current estimate for a period and overwrites it as
that estimate changes. Forecast accuracy at a given lead time therefore cannot
be measured after the fact from the API alone. This repository records each
forecast at the moment it is issued, records the realised value when it settles,
and scores one against the other.

The eventual aim is a learned correction to the published forecast, promoted
into use only when it beats the published forecast out of sample. Nothing of
that exists yet; the current state is described below.

## Status

Working: the API client, the parsers, schema validation, raw snapshot storage,
the command-line interface, and the scheduled capture workflow.

Not yet built: feature construction, the baseline and candidate models, the
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
| `/generation` | Fuel shares for the period in progress |

Raw responses are written verbatim under `data/raw/YYYY/MM/DD/`, each wrapped in
an envelope recording the endpoint requested and the time the response arrived.
Filenames are built from a short validated label rather than the request path,
because request paths contain colons and a colon in a Windows path silently
opens an alternate data stream instead of creating a visible file.
Parsed derivatives are not stored. The parser will change; a stored payload can
be reprocessed under a corrected parser, and a stored derivative cannot.

## Two conventions worth knowing

**Capture time is part of the data.** Every parsed row carries `captured_at` and
a `horizon_hours` lead time. Without them a forecast is indistinguishable from a
later revision of itself, which makes accuracy by lead time unmeasurable.

**The live API does not match its published schema.** `/generation` returns a
bare object where the specification promises an array. Both parsers normalise
either form. Because the offline fixtures are recorded from the specification,
they cannot catch this class of divergence; the daily contract workflow exists
for that.

**Half-hour positions are UTC, not settlement periods.** GB settlement periods
are numbered against the local clock day, which has 46 or 50 of them at the
daylight-saving transitions. The parser labels rows `utc_half_hour` and carries
local clock time separately. The two agree for most of the year and disagree
exactly on the days when the system behaves unusually.

## Use

```
pip install -e ".[dev]"

gridcast snapshot                                        # one capture
gridcast backfill --start 2024-01-01 --end 2026-01-01    # historical actuals
```

Backfill splits its range into fourteen-day windows, which is the maximum the
range endpoint accepts.

## Tests

```
pytest -m "not network"     # offline, against recorded fixtures
pytest -m network           # exercises the live API
```

Unit tests run against fixtures recorded from the documented response schemas,
so the suite passes without network access. The network-marked tests check that
the live API still returns the shape the parsers assume; they run in CI on a
schedule rather than on every commit.

## Prior art

[nmpowell/carbon-intensity-forecast-tracking](https://github.com/nmpowell/carbon-intensity-forecast-tracking)
scrapes the same API on a schedule to compare forecasts against realised values.
This repository takes the same starting point and goes on to model the residual
and gate a candidate model on beating the published forecast.

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
against the live API, the loader that joins issued forecasts to the outcomes
they were forecasting, two seasonal baselines, the scoring harness, feature
construction, the scheduling and regret evaluation, the revision analysis, a
damped-revision correction, and the promotion gate that keeps checking it.

Not built: drift monitoring and the published evaluation report.

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
| `/generation/{from}/{to}` | Fuel shares for the past 24 hours |

Capture is scheduled hourly and delivered far less often: GitHub queues
scheduled workflows at low priority and drops them under load. Delivery fell
from around 38 runs a day in the first week to 2 a day in the second, and
reducing the nominal cadence from half-hourly to hourly did not change it.
Because each run re-harvests the past day, a missed run costs forecast vintages
but no outcomes, so the outcome record is complete while the vintage record is
not. The revision and correction analyses therefore rest on 20-26 August, when
capture was dense. The measurement and what was ruled out are in
`docs/design.md`.

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
of both intensity and generation mix, rather than only the period in progress.
A run asking only about the present loses any period whose run was delayed or
dropped, and loses it permanently. Since the scheduler delivers roughly 60% of
its nominal half-hourly runs, that is a continuous loss rather than an
occasional one. Capturing the past day means a period is reported by 48
successive runs.

**Availability is decided by when a value could have been known.** A backfilled
observation carries the capture time of the backfill run, which is today. Using
that as the test of what a prediction may see would declare two years of settled
history unavailable to any forecast issued before the backfill, and the matched
comparison would silently score nothing. A period is instead treated as
available once it has ended and a one-hour settlement allowance has passed.

**The unit of prediction is a period and an issue time, not a period.** A
prediction thirty minutes ahead may lean on an observation made an hour ago; one
forty-eight hours ahead may not. The training set is therefore the cross product
of observed periods with a set of hypothetical lead times, features recomputed
at each, so that no row sees anything that did not exist when its prediction was
supposedly made. Rows sharing a period are correlated, so validation splits by
date rather than by row.

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
gridcast compare                                         # forecast against baselines
gridcast schedule --periods 4 --window 24                # score decision quality
gridcast audit --periods 4 --window 24                   # inspect that comparison
gridcast revisions                                       # how forecasts move over time
gridcast correct                                         # test a damped-revision correction
gridcast gate                                            # check the correction still holds
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

## Decision quality

`gridcast schedule` poses a concrete decision: a deferrable load of a given
length must run within a given window, and the scheduler picks the periods a
forecast says are cheapest. Four outcomes are costed against what actually
happened — the choice made, the choice hindsight would have made, the worst
choice available, and running immediately without deferring at all.

Regret is the excess of the first over the second. Reported alone it misleads,
because on a flat day no choice is much worse than any other and a windless week
would flatter any scheduler. It is normalised by the saving that was available,
giving the fraction of the achievable benefit the forecast secured.

The table has three kinds of row. `published` and the `_matched` baselines face
the same decisions — the issue times where a forward snapshot was captured, and
the same windows — so differences between those rows are differences between
forecasters. The `_full` rows score the baselines across the whole settled
record: a much larger sample, and not comparable with the first two, because the
captured decisions come from a few weeks of one season and the windows differ in
how much saving was available at all. Compare `mean_available` before comparing
anything else.

## Forecast revisions

Each period is forecast repeatedly as it approaches, and the API keeps no
history of those forecasts, so the sequence for a given period is something this
project holds and the source cannot reproduce. `gridcast revisions` asks three
things of it.

Whether accuracy improves with proximity, measured on periods forecast at every
lead so that horizon rather than weather is what differs between the bands.

Whether successive revisions are correlated. A forecast using all available
information should revise unpredictably: the change from one issue to the next
should say nothing about the change after it. Positive correlation would mean it
adjusts gradually towards news it has already received; a value near minus a
half would mean it jitters around a level rather than converging, retracing most
of its own movement.

Consecutive captures that found no change are collapsed first. Roughly half of
captures see an unchanged value, and those repeats dilute any correlation
towards zero. Collapsing them also gives a figure the API does not publish: how
often the forecast is actually revised, by lead time.

Whether a revision anticipates the error still remaining, which would be
directly exploitable and is the sharper version of the same question.

## A damped-revision correction

If the published forecast overshoots — revising further than the outcome
justifies — then part of its most recent revision should be subtracted rather
than believed:

    corrected = published - damping * most_recent_revision

One coefficient per lead band, obtained in closed form as the least-squares
slope of the remaining error on the revision, so nothing is tuned by hand. A
positive coefficient means the forecast overshoots; a negative one would mean it
under-reacts and the revision should be amplified instead.

Fitted on earlier dates and scored on later ones, split by date rather than by
row because revisions of the same target period are not independent. Bands with
fewer than thirty observations are left uncorrected rather than fitted on noise.

Each improvement carries a 95% interval from a paired bootstrap resampled by
target period. A band is worth believing only where the lower bound is above
zero; a point estimate on its own cannot distinguish a real gain from none.

## The promotion gate

A correction is not established by having been significant once. The coefficient
was fitted on four days of late August, and whether it describes the published
forecast or that week's weather is a question only time answers.

`gridcast gate` refits on the current record and promotes a band only if the
improvement's interval excludes zero, the band carries enough held-out
observations and enough distinct target periods, and the refitted coefficient is
close to the one it replaces. That last condition is the one an interval cannot
supply: a coefficient swinging from 0.46 to 0.93 between refits describes the
sample rather than the forecast, however tight its interval.

Every run's fitted coefficient is kept, promoted or not, so that stability
across refits is visible: one run's interval speaks only to that run's sample,
and a coefficient settling near one value over time is evidence of a different
kind.

The train-test split divides on the date that balances rows rather than the date
at a fixed position, because this record's days are wildly uneven in size and a
date-position split makes every result depend on the capture schedule.

The verdict is stored in `docs/promoted.json` and the workflow runs weekly. It
fails the build only when a band that *had* been promoted no longer qualifies —
including one that disappears from the evaluation entirely, which leaves its
recorded coefficient standing on nothing. A band that has never qualified
failing again is the normal state, and failing the build for that would make it
red from the first run and teach everyone to ignore it.

## Does the correction help the decision?

The correction is validated on mean absolute error. This project's whole premise
is that accuracy is not what a scheduler consumes, so the correction has to face
its own test: `gridcast schedule` scores a `corrected` row beside the published
forecast, on identical decisions, using whatever coefficient the gate has
promoted. An improvement in accuracy that does not survive into decision quality
would be a result rather than an embarrassment — it is exactly what the opening
claim of this repository predicts is possible.

Only promoted coefficients are applied. Correcting with a coefficient the gate
has held back would be scoring a claim the project does not make.

## Known gaps in the record

A single 16-hour outage in the intensity series on 12 June 2024. Everything else
is continuous from 31 December 2023.

The generation mix is absent for roughly the first twelve days of January in
both 2025 and 2026 — contiguous blocks beginning within half an hour of the new
year, while the intensity series continues uninterrupted. Two occurrences at the
same calendar position suggest a property of the source rather than a fetch
fault. Any evaluation over a January must be conditioned on it.

Gaps are recorded, never interpolated. A fabricated observation cannot be
distinguished from a real one downstream, and a model trained partly on invented
data cannot be evaluated honestly.

## Prior art

[nmpowell/carbon-intensity-forecast-tracking](https://github.com/nmpowell/carbon-intensity-forecast-tracking)
scrapes the same API on a schedule to compare forecasts against realised values.
This repository starts from the same measurement problem; the intended addition
is the learned correction and promotion gate described above.

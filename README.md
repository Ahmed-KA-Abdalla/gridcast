# gridcast

[![ci](https://github.com/Ahmed-KA-Abdalla/gridcast/actions/workflows/ci.yml/badge.svg)](https://github.com/Ahmed-KA-Abdalla/gridcast/actions/workflows/ci.yml)
[![contract](https://github.com/Ahmed-KA-Abdalla/gridcast/actions/workflows/contract.yml/badge.svg)](https://github.com/Ahmed-KA-Abdalla/gridcast/actions/workflows/contract.yml)
[![gate](https://github.com/Ahmed-KA-Abdalla/gridcast/actions/workflows/gate.yml/badge.svg)](https://github.com/Ahmed-KA-Abdalla/gridcast/actions/workflows/gate.yml)

Measuring the Great Britain carbon intensity forecast by the decisions it
produces rather than by its error.

Nobody consumes carbon intensity for its own sake. It is used to decide when to
run something: charge a car, heat water, run a wash. Forecast accuracy is a
proxy for that, and the two come apart. A forecast uniformly too high by 40
gCO2/kWh has a mean absolute error of 40 and schedules perfectly, because adding
a constant changes no ordering. A forecast accurate to 8 that inverts the two
cheapest windows schedules badly. Only the ordering matters, and mean error does
not measure ordering.

The forecast is published by the National Energy System Operator and revised
every half hour, and no history of those revisions is kept: the API serves the
current estimate for a period and overwrites it. This repository has been
recording each forecast as it is issued since 20 August 2026, alongside the
realised values, so that the two can be scored against each other.

## What it found

All figures below are for a two-hour deferrable load with twenty-four hours of
slack, over 184 to 198 captured decisions in late August 2026. The samples are
small and one season only; the last section says what else is missing.

**Deferring works, and the published forecast captures nearly all of the
benefit.** Scheduling on the forecast rather than running immediately saved 39.5
gCO2/kWh averaged over the load, around 15% of a typical GB intensity. Against
perfect hindsight it secured 92.5% of the saving that was available.

**Most of that benefit comes from the daily shape, not from the weather.** A
seasonal baseline — the mean intensity at the same half-hour of the last three
same weekdays, which knows nothing about wind — captured 94.6% on the same
decisions. It is not that the published forecast is poor: it is three times more
accurate than the baseline by mean absolute error. It is that accuracy in excess
of the daily and weekly pattern buys very little for this decision.

**The published forecast overshoots at medium lead.** Successive revisions are
anticorrelated at -0.51, and a revision is negatively correlated with the error
still remaining at -0.50 in the six-to-twelve-hour band. Median total movement
of a forecast is 117.5 gCO2/kWh against median net movement of 12, so most of
what it does is later undone. Subtracting 46% of the most recent revision cuts
mean absolute error in that band by about 5%, out of sample: an improvement of
1.05 gCO2/kWh with a bootstrap interval of +0.65 to +1.40, over 571 held-out
observations across 186 target periods. The coefficient has come out at 0.46,
0.46 and 0.47 across three refits on different splits.

**That improvement changes no decisions.** Scheduled on identical decisions, the
corrected forecast chose the same window every time for a contiguous load — hit
rate identical to three decimals. For an interruptible load, which can pick
individual periods rather than a block, it changed a few and made them slightly
worse, moving the hit rate from 0.338 to 0.323. Damping subtracts a similar
amount from every period in a window, and a near-uniform shift changes little
ordering.

Taken together: for a carbon-aware scheduler, effort spent on more accurate
intensity forecasting is not where the remaining value is. Effort spent on
giving loads more slack is — a six-hour window offers only 38 gCO2/kWh of
available saving against 91 at twenty-four hours.

**Four of five lead bands showed no correction at all.** Only six-to-twelve
hours survives the promotion gate. The three-to-six-hour band cleared on 28
August and no longer does, which is what a borderline effect looks like when the
split moves.

## Status

Built: the API client, parsers, schema validation, raw snapshot storage, the
command-line interface, the scheduled capture workflow, a daily contract check
against the live API, the loader joining issued forecasts to outcomes, two
seasonal baselines, the scoring harness, feature construction, the scheduling
and regret evaluation, the revision analysis, the damped-revision correction,
and the promotion gate that keeps checking it.

Not built: drift monitoring and a published evaluation page.

## Data

Source: the NESO Carbon Intensity API, `https://api.carbonintensity.org.uk`,
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). No API key
is required. The API definition is at
<https://carbon-intensity.github.io/api-definitions/>.

Five endpoints are captured on each run:

| Endpoint | Purpose |
| --- | --- |
| `/intensity` | The period in progress |
| `/intensity/{from}/fw48h` | The forecast as issued, out to 48 hours |
| `/intensity/{from}/pt24h` | Realised values for the past 24 hours |
| `/generation` | Fuel shares for the period in progress |
| `/generation/{from}/{to}` | Fuel shares for the past 24 hours |

Raw responses are written verbatim under `data/raw/YYYY/MM/DD/`, wrapped in an
envelope recording the endpoint and the time the response arrived. Parsed
derivatives are not stored: the parser will change, and a stored payload can be
reprocessed under a corrected parser where a derivative cannot.

Capture is scheduled hourly and delivered erratically, because GitHub queues
scheduled workflows at low priority and drops them under load. Delivery ran at
around 38 runs a day in the first week, fell to 2 a day in the second, and has
since recovered to 4 or 5. Reducing the nominal cadence from half-hourly to
hourly made no difference. Because each run re-harvests the past day, a missed
run costs forecast vintages but no outcomes, so the outcome record is complete
while the vintage record is not.

The outcome record holds 46,540 settled half-hours from 31 December 2023, with
31 missing.

## Conventions worth knowing

**The unit of prediction is a period and an issue time, not a period.** A
prediction thirty minutes ahead may lean on an observation made an hour ago; one
forty-eight hours ahead may not. Features are recomputed at each lead, and rows
sharing a period are correlated, so splits are by date rather than by row.

**Only forecasts captured at issue time count as forecasts.** The `forecast`
field of a historical range response is a revised value, produced with
information unavailable at the lead time it appears to occupy. Using it would
leak the outcome into the predictor without disturbing any metric.

**Availability is decided by when a value could have been known.** A backfilled
observation carries the capture time of the backfill run. Testing against that
would declare two years of settled history unavailable to any earlier forecast.
A period is available once it has ended and a one-hour settlement allowance has
passed.

**Half-hour positions are UTC, not settlement periods.** GB settlement periods
are numbered against the local clock day, which has 46 or 50 of them at the
daylight-saving transitions.

**Filenames come from a validated label, not the request path.** Request paths
contain colons, and on NTFS a colon separates a filename from an alternate data
stream, so writing to such a path succeeds and produces a file no directory
listing shows.

**The live API does not match its published schema.** `/generation` returns a
bare object where the specification promises an array. Fixtures recorded from a
specification cannot detect that the specification and the service disagree,
which is what the daily contract check is for.

## Method

`gridcast schedule` poses a concrete decision: a load of a given length must run
within a given window, and the scheduler picks the periods a forecast says are
cheapest. Four outcomes are costed against realised intensity — the choice made,
the choice hindsight would have made, the worst available, and running
immediately. Regret is the first minus the second, normalised by the available
spread, since on a flat day no choice is much worse than any other and an
unnormalised figure would reward calm weather.

The published forecast and the baselines are scored on identical decisions,
because the sample matters more than it appears to: an earlier version compared
the published forecast against baselines scored across the whole record, and the
difference in available spread was large enough to reverse the ranking.

`gridcast correct` fits one damping coefficient per lead band as the
least-squares slope of the remaining error on the revision, in closed form, on
earlier dates, and scores it on later ones. Each improvement carries a 95%
interval from a paired bootstrap resampled by target period.

`gridcast gate` refits and promotes a band only where the interval excludes
zero, the band carries enough held-out observations and distinct periods, and
the refitted coefficient is close to the one it replaces. That last condition is
the one an interval cannot supply. Every run's coefficient is kept, promoted or
not, so stability across refits is visible. The gate fails the build only when a
band that had been promoted no longer qualifies.

The train-test split divides on the date that balances rows rather than the date
at a fixed position, because this record's days are wildly uneven in size and a
date-position split makes every result depend on the capture schedule.

`docs/design.md` records the reasoning behind each of these, including the
faults found along the way and what was ruled out.

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
gridcast correct                                         # test the correction
gridcast gate                                            # check it still holds
```

## Tests

```
pytest -m "not network"     # offline, against recorded fixtures
pytest -m network           # exercises the live API
```

240 tests, 96% line coverage. The offline suite needs no network. The
network-marked tests check that the live API still returns the shape the parsers
assume, and run daily rather than on every commit.

## What is missing

One season. Everything here comes from late August 2026. The correction's
coefficient may be a property of the forecast or of that fortnight's weather,
and only time separates them — which is what the weekly gate exists to find out.

Small decision samples. Under 200 captured decisions, from one load shape at a
time.

The generation mix is absent for roughly the first twelve days of January in
both 2025 and 2026, in contiguous blocks beginning within half an hour of the
new year, while the intensity series continues. Two occurrences at the same
calendar position suggest a property of the source. A single 16-hour intensity
outage on 12 June 2024; everything else is continuous.

Gaps are recorded, never interpolated. A fabricated observation cannot be
distinguished from a real one downstream.

## Prior art

[nmpowell/carbon-intensity-forecast-tracking](https://github.com/nmpowell/carbon-intensity-forecast-tracking)
scrapes the same API on a schedule and publishes daily accuracy statistics. It
was found before this project began and its capture design was arrived at
independently, which is worth saying plainly: the two converge on the same
approach to a problem the API's lack of history forces on anyone measuring it.
That project measures forecast accuracy. This one measures the decision quality
accuracy is a proxy for, and finds the two do not track each other.

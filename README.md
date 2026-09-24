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

All figures are for a two-hour deferrable load with twenty-four hours of slack,
over 322 decisions the published forecast faced between 20 August and 23
September 2026, unless stated otherwise. Figures were last refreshed on 24
September; the capture and gate workflows keep running, so a fresh run may
disagree with what is written here.

Several of these numbers have moved as the sample grew, and two reversed. Where
that happened it is stated, because a result that changes with a third more data
is one to hold loosely.

**Deferring works.** Scheduling on the published forecast rather than running
immediately saved 38.3 gCO2/kWh averaged over the load, around 15% of a typical
GB intensity. Against perfect hindsight it secured 90.7% of the saving that was
available.

**The published forecast beats a seasonal baseline on decisions.** A baseline
taking the mean intensity at the same half-hour of the last three same weekdays
— which knows nothing about wind — secured 86.7% on the same decisions, against
the forecast's 90.7%. On a smaller sample of 195 decisions this was the other
way round, with the baseline ahead at 93.6% against 92.4%. The reversal is the
clearest illustration in this repository of how far a result can move on two
hundred decisions.

**Most of the achievable benefit needs no weather information.** The gap between
the two forecasters is four percentage points; the gap between doing nothing and
deferring is fifteen per cent of intensity. The daily and weekly pattern carries
most of what a scheduler needs.

**The published forecast overshoots at short and medium lead.** Successive
revisions are anticorrelated: -0.32 in the zero-to-three-hour band, -0.38 in
three-to-six, -0.54 in six-to-twelve, and near zero beyond twelve hours where
the forecast is barely revised at all. Subtracting a fitted share of the most
recent revision reduces error out of sample — 1.01 gCO2/kWh in the
zero-to-three-hour band with a bootstrap lower bound of +0.36, and 1.87 in
three-to-six with a lower bound of +1.17.

**How much a forecast moves depends on how often you look.** An earlier version
of this file reported median total movement of 117.5 gCO2/kWh against median net
movement of 12, and concluded that most of what the forecast does is later
undone. Both figures are sampling artefacts: they were measured when capture ran
about fifty times per period, and now read 48.0 and 13.0 at about eleven. The
anticorrelation is the sampling-robust version of the same observation, and it
is what the correction rests on.

**Which band survives has moved.** The six-to-twelve-hour band cleared the
promotion gate on three successive refits and no longer does. The short bands
now clear instead. The fitted coefficients have been stable throughout — 0.40 to
0.46 across every band under twelve hours, over five refits — so what moves is
which band reaches significance, not the size of the effect.

**The correction's effect on decisions is small.** The corrected forecast
secured 91.5% against the published forecast's 90.7%, with mean regret 7.40
against 7.86. On an earlier sample it changed no decisions at all. There is no
interval on this comparison, so it is reported and not claimed. Damping
subtracts a similar amount from every period in a window, and a near-uniform
shift changes little ordering.

**A model trained on the level improves decisions significantly.** Gradient
boosting over the leak-safe features, fitted on 597 generated decisions from
2024 to mid-2025 and scored on 398 from the following year, reduced mean regret
by 1.91 gCO2/kWh against the seasonal baseline, with a paired interval of +0.43
to +3.40. This was not expected: the working hypothesis was that a model trained
on squared error would improve accuracy without reaching the decision.

**The target matters; the loss function does not.** Three objectives were fitted
on the same features, the same 597 training decisions, and scored on the same
398 held-out ones. Squared error on the level secured 88.6% of the available
saving. Squared error on each period's deviation from its own window mean
secured 90.2%, reducing mean regret by 1.17 gCO2/kWh against the level model
with an interval of +0.20 to +2.20. A pairwise ranking loss, learning only which
of two periods is cheaper, did worse than the level model at -1.85.

The pairwise model is linear, because a scorer fitted on feature differences
must be additive to induce an ordering on single periods, while the others are
gradient-boosted trees. A linear model on the relative target was therefore
fitted as a control, and the pairwise model is indistinguishable from it: -0.31
with an interval of -1.40 to +0.66. So the pairwise deficit is its functional
form, not its objective. Against a like-for-like control, the ordering objective
buys nothing.

## Status

Built: the API client, parsers, schema validation, raw snapshot storage, the
command-line interface, the scheduled capture workflow, a daily contract check
against the live API, the loader joining issued forecasts to outcomes, two
seasonal baselines, the scoring harness, feature construction, the scheduling
and regret evaluation, the revision analysis, the damped-revision correction,
the promotion gate that keeps checking it, a decision dataset over the whole
settled record, a gradient-boosting model scored on both accuracy and decision
quality, three training objectives compared against each other, and drift monitoring.

Not built: a published evaluation page.

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

The outcome record holds 47,824 settled half-hours from 31 December 2023, with
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

`gridcast gate` refits and promotes a band only where the improvement's lower
bound clears a minimum worth claiming, the band carries enough held-out
observations and distinct periods, and the refitted coefficient is close to the
one it replaces. The bound is a tenth of a gCO2/kWh rather than zero: a
coefficient near zero applies almost no correction, so its improvement is tiny
and its variance is tiny with it, and the interval clears zero while the effect
is nothing. One band was promoted that way before the threshold was added. That last condition is
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
gridcast decisions --periods 4 --window 24               # the decision dataset
gridcast model --periods 4 --window 24                   # fit and score a model
gridcast objectives --periods 4 --window 24               # compare objectives
gridcast drift                                           # has the data moved?
```

## Tests

```
pytest -m "not network"     # offline, against recorded fixtures
pytest -m network           # exercises the live API
```

314 tests, 95% line coverage. The offline suite needs no network. The
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

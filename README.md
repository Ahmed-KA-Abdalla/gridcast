# gridcast

[![ci](https://github.com/Ahmed-KA-Abdalla/gridcast/actions/workflows/ci.yml/badge.svg)](https://github.com/Ahmed-KA-Abdalla/gridcast/actions/workflows/ci.yml)
[![contract](https://github.com/Ahmed-KA-Abdalla/gridcast/actions/workflows/contract.yml/badge.svg)](https://github.com/Ahmed-KA-Abdalla/gridcast/actions/workflows/contract.yml)
[![gate](https://github.com/Ahmed-KA-Abdalla/gridcast/actions/workflows/gate.yml/badge.svg)](https://github.com/Ahmed-KA-Abdalla/gridcast/actions/workflows/gate.yml)

**[Full evaluation page →](https://ahmed-ka-abdalla.github.io/gridcast/)**

<!-- figures:start -->

_Figures below regenerated 2026-10-27 from 49,449 settled half-hours and 503 captured forecast issues._

**Decision quality** — 2h contiguous load within 24h. Rows ending `_matched` face the same
decisions as the published forecast; `_full` rows are a different sample.

| forecaster             |    n |   mean_regret |   hit_rate |   captured_fraction |
|:-----------------------|-----:|--------------:|-----------:|--------------------:|
| published              |  487 |         8.767 |      0.187 |               0.893 |
| seasonal_naive_matched |  487 |        17.794 |      0.086 |               0.814 |
| seasonal_mean_matched  |  487 |        15.803 |      0.07  |               0.84  |
| seasonal_naive_full    | 1022 |        17.535 |      0.073 |               0.806 |
| seasonal_mean_full     | 1022 |        12.77  |      0.082 |               0.856 |

**The promoted correction, out of sample** (asymmetric damping of (3, 6], (6, 12]): mean regret 10.883
published against 9.370 corrected over 236 held-out decisions, a gain of +1.514 (+0.349 to +2.749), helped in 5 of 5 blocks.

**Damping correction**, as the gate last judged it: separate coefficients for upward and downward revisions, scored on 5 consecutive
held-out blocks, each fitted on every earlier date.

| band     | verdict   |     up |   down |   improvement |   interval low |   interval high | blocks helped   |    n |
|:---------|:----------|-------:|-------:|--------------:|---------------:|----------------:|:----------------|-----:|
| (0, 3]   | held back |  0.123 |  0.579 |        -0.001 |         -0.38  |           0.391 | 2 of 5          | 1386 |
| (12, 24] | held back | -0.311 |  1.498 |         0.712 |          0.543 |           0.88  | 3 of 5          | 4906 |
| (24, 48] | held back | -0.551 |  1.275 |        -0.595 |         -0.783 |          -0.407 | 3 of 5          | 7587 |
| (3, 6]   | promoted  |  0.184 |  0.714 |         0.954 |          0.578 |           1.341 | 4 of 5          | 1369 |
| (6, 12]  | promoted  |  0.037 |  0.86  |         1.592 |          1.301 |           1.886 | 5 of 5          | 2614 |

**Seasonal drift** of intensity against summer, which is when the captured
forecasts begin.

| season   |   psi_against_summer | reading        |    mean |     n |
|:---------|---------------------:|:---------------|--------:|------:|
| winter   |                0.285 | large shift    | 143.097 | 11521 |
| spring   |                0.037 | stable         | 118.303 | 13248 |
| summer   |                0     | stable         | 114.307 | 13217 |
| autumn   |                0.111 | moderate shift | 128.144 | 11463 |

<!-- figures:end -->

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
September, except those for the correction, refreshed on 27 October; the
capture and gate workflows keep running, so a fresh run may disagree with what
is written here.

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
recent revision reduces error out of sample, provided upward and downward
revisions are treated separately.

**How much a forecast moves depends on how often you look.** An earlier version
of this file reported median total movement of 117.5 gCO2/kWh against median net
movement of 12, and concluded that most of what the forecast does is later
undone. Both figures are sampling artefacts: they were measured when capture ran
about fifty times per period, and now read 48.0 and 13.0 at about eleven. The
anticorrelation is the sampling-robust version of the same observation, and it
is what the correction rests on.

**The overshoot is asymmetric.** A downward revision is mostly retraced; an
upward one mostly is not. Fitted separately, the coefficient on downward
revisions is 0.71 at three to six hours and 0.86 at six to twelve, and on upward
revisions 0.18 and 0.04. One coefficient for both directions, which is what the
correction used until October, splits the difference and is wrong for both. It
was promoted at short lead in September and was significantly harmful there by
October, once autumn brought a run of large upward revisions.

Scored on five consecutive held-out blocks from 11 September to 27 October, each
corrected with coefficients fitted on every earlier date, the asymmetric
correction reduces mean absolute error by 1.59 gCO2/kWh at six to twelve hours
(+1.30 to +1.89), helping in all five blocks, and by 0.95 at three to six
(+0.58 to +1.34), helping in four. The symmetric correction manages 0.26 and
0.52 on the same blocks. Under three hours neither helps. Beyond twelve hours the
asymmetric coefficients swing between refits — the upward one from -0.50 to
+2.31 — so nothing there is claimed, although the pooled interval at twelve to
twenty-four hours looks clean on its own.

**The asymmetric correction improves decisions out of sample.** Applied to the
same five blocks with coefficients fitted only on earlier dates, it reduced mean
regret on the two-hour load from 10.88 to 9.37 gCO2/kWh over 236 decisions on 46
days: a gain of 1.51 with an interval of +0.35 to +2.75, resampling whole days,
positive in all five blocks. It changed 17% of decisions. The symmetric
correction it replaces did not help decisions (-0.45, -1.16 to +0.14), and for a
one-hour load within six hours the asymmetric gain is not significant (+0.31,
-0.42 to +1.14). The bands applied were chosen by the gate on the same record,
so their selection is informed by these blocks even though no coefficient is.
An earlier version of this file said the correction changed almost no decisions;
that described the symmetric correction scored in sample.

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
quality, three training objectives compared against each other, drift monitoring, and a
published evaluation page rebuilt weekly from the record.

Everything planned is built.

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

`gridcast correct` fits, for each lead band, one damping coefficient for
upward revisions and one for downward, as the least-squares fit of the
correction actually applied: `corrected = published - damping * revision`, with
no intercept. It scores the result on five consecutive held-out blocks, of
roughly equal rows rather than equal days because capture has been uneven, each
corrected with coefficients fitted on every earlier date and purged of the
target periods the block scores, and reports the symmetric model alongside on
the same blocks. Each improvement carries a 95% interval from a paired bootstrap
resampled by target period.

`gridcast gate` promotes a band only where the pooled improvement's lower bound
clears a tenth of a gCO2/kWh, the band carries enough held-out observations and
distinct periods, the correction helped in at least four blocks of five, and
each coefficient moved by no more than 0.25 between the blocks' fits and against
the last run's. The bound is a tenth rather than zero because a coefficient near
zero applies almost no correction, so its interval clears zero while the effect
is nothing; one band was promoted that way before the threshold was added. The
block and stability checks exist because a single train-test split was not
enough: the symmetric correction at short lead was significant on one split and
significantly harmful on the next.

The gate fails the build only when a band that had been promoted no longer
qualifies under the same model, and records the withdrawal on the run that
fails, so the next run starts from the corrected record. The commit step
originally ran only on success, so a failing gate computed the withdrawal and
discarded it, then failed identically every week after. A change of model is
not a regression: the record names its model, and the bands a new model drops
are listed when it takes over.

`gridcast schedule` reports the promoted correction's effect on decisions out of
sample, on the same held-out blocks. It previously scored the corrected forecast
beside the published one on the whole record, with coefficients fitted on that
record, which flattered it.

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
gridcast report-page --out site/index.html               # build the evaluation page
gridcast refresh-readme                                  # update the figures above
```

## Tests

```
pytest -m "not network"     # offline, against recorded fixtures
pytest -m network           # exercises the live API
```

360 offline tests and one network test, 94% line coverage. The offline suite
needs no network. The
network-marked tests check that the live API still returns the shape the parsers
assume, and run daily rather than on every commit.

## What is missing

One autumn. Captured forecasts begin on 20 August 2026, and the correction has
been tested across the turn from summer into autumn but not into winter, which
is a larger shift in the intensity distribution. Whether the asymmetry holds is
what the weekly gate exists to find out.

Small decision samples. Under 500 captured decisions, from one load shape at a
time, and 236 of them held out.

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

# Design note

This records the decisions taken in the capture layer and the reasoning behind
them, so that later work does not silently reverse one.

## Problem

NESO publishes a forecast of GB carbon intensity out to 48 hours and revises it
every half hour. The API serves the current estimate for each settlement period
and overwrites it as the estimate changes; no history of superseded forecasts is
kept. A forecast's accuracy at a given lead time is therefore not recoverable
after the fact.

Two consequences follow. Measuring the published forecast requires capturing it
as it is issued, and any correction model trained on the historical `forecast`
field would be trained on values revised with information the model would not
have at prediction time. The second is a leakage route that would not announce
itself: the model would look accurate and would fail in use.

## Decisions

**Capture at issue time, on a schedule.** The forward forecast is fetched every
half hour and stored with the time of capture. This is why the project runs on a
schedule rather than on demand: the record can only be built forwards.

**Name files from a short label, not the request path.** Request paths contain
colons. On NTFS a colon separates a filename from an alternate data stream, so
writing to such a path succeeds, reports success, and produces a file that no
directory listing shows. Snapshots are therefore named from a validated `kind`
label matching `[a-z0-9_]+`, with the full request path recorded in the
envelope where its punctuation is harmless. The label is validated rather than
sanitised: an unexpected value should fail at the call site instead of being
silently rewritten into something else.

**Store raw payloads, not parsed frames.** Responses are written verbatim inside
an envelope holding the endpoint, the capture time, and the licence. The parser
will change over the life of the project. A stored payload can be reprocessed
under a corrected parser; a stored derivative cannot, and the record is
irreplaceable.

**Label half-hour positions in UTC.** GB settlement periods are numbered against
the local clock day, which has 46 periods in March and 50 in October. Numbering
UTC half-hours 1-48 and calling them settlement periods would be correct for
363 days a year and wrong on the two when the system is least ordinary. Rows
carry `utc_half_hour` for the position on the UTC grid and `local_hour` for the
clock time that drives demand behaviour.

**Validate before storing.** Each payload is parsed and checked before it is
written: half-hour grid spacing, no duplicated or unordered periods, intensity
within 0-1000 gCO2/kWh, fuel shares summing to 100 per cent within the rounding
tolerance the API's one-decimal publication implies. A failing endpoint is
skipped and the run exits non-zero; the other endpoints in the same run are
still stored, since one malformed response is not a reason to lose the others.

**Treat an absent fuel as zero, not missing.** The API omits a fuel from the mix
when its share is nil. A missing share and an unknown share are different facts
and only one of them is true here.

**Split backfill into fourteen-day windows.** The range endpoint refuses
anything longer. Backfill recovers realised values only; it cannot recover
forecasts as issued, for the reason given above.

**Normalise the `data` field before parsing.** The published schema states that
`data` is always an array. The live `/generation` endpoint returns a bare object
for the current period. Iterating that object yields its keys, so the deviation
surfaced as a type error inside a parser rather than as a clear failure at the
boundary. Both parsers now go through a normaliser that accepts either form and
raises `ParseError` on anything else. This is the case for the daily contract
test: fixtures recorded from a published schema cannot detect that the schema
and the service disagree.

**A bad payload costs one endpoint, not the run.** Fetch failures and parse or
validation failures are caught per endpoint. The remaining endpoints are still
captured and the process exits non-zero, so a run that captured two of three is
visibly degraded rather than either silently partial or entirely lost.

**Harvest outcomes with redundancy, not just the current period.** The first
version of the capture step fetched only the period in progress alongside the
forward forecast. That is one outcome per run against 96 forecast rows, and a
period whose run was delayed or dropped is lost for good, since backfill of
recent days is not indefinitely available. Each run now fetches the past 24
hours, so every period is reported by 48 successive runs.

## Baselines and scoring

Two seasonal baselines: the value at the same half-hour of the most recent
available same weekday, and the mean across three such weeks. Both fall back
through lags of 7, 14 and 21 days, so a gap in the record costs a longer lag
rather than a missing prediction. Which is harder to beat is an empirical
question, which is why both exist.

Two evaluations, answering different questions. A seasonal prediction does not
depend on lead time, so it can be scored over every period ever observed, which
yields a usable number immediately. The published forecast can be scored only
where it was captured. Reporting one in place of the other would flatter
whichever is favoured by the difference in sample, so both are printed and
labelled.

Availability had to be defined carefully. The first implementation asked whether
a reference observation had been captured before the moment of prediction. That
is wrong for backfilled data: a backfill run stamps two years of settled history
with today's capture time, so every historical reference would have been treated
as unknown to any forecast issued earlier, and the matched comparison would have
scored nothing at all while appearing to work. Availability is now decided by
period timing — a period is knowable once it has ended and a one-hour settlement
allowance has passed — which is a property of the data rather than of this
project's fetch history.

## Features

Every feature is computed as of an explicit issue time, and the training set is
the cross product of observed periods with nine hypothetical lead times from
half an hour to forty-eight. A single "recent intensity" column applied without
regard to lead would be using tomorrow's data to predict tomorrow.

Calendar position is encoded as sine and cosine pairs on the daily and annual
cycles, so that 23:30 and 00:00 are adjacent rather than maximally distant. The
daily cycle is taken in local clock time, which is what drives demand behaviour,
while the index remains UTC.

The most recent knowable intensity and fuel mix come from a backward as-of join
on the settlement time. One expectation was wrong here and the measurement
corrected it: the staleness of that observation, measured from the issue time,
was assumed to grow with lead time and does not. There is almost always an
observation that has just become knowable, so staleness sits at the settlement
allowance regardless of lead. It is retained because its exceptions are
informative — it rises above the floor exactly where the record has a gap — but
not for the reason it was added.

**The generation mix needed the same redundancy as intensity, and did not get it
at first.** The day-two fix added a past-24-hour fetch for intensity but left
generation fetching only the period in progress. With the scheduler delivering
about 60% of nominal runs, that lost two or three mix periods for every one
captured. It surfaced when the fuel-mix features came out missing at 2.55%
against 0.07% for intensity — a discrepancy too large to be scattered gaps. The
lesson is narrow but worth recording: a fix applied to one path in a
multi-endpoint capture is not a fix.

The same investigation turned up a genuine source characteristic. The mix is
absent for roughly the first twelve days of January in both 2025 and 2026, in
contiguous blocks starting within half an hour of the new year, while intensity
continues uninterrupted. Two occurrences at the same calendar position are not
coincidence, but the cause is on NESO's side and nothing here can recover it.

## Scheduling and regret

The decision, not the prediction, is the unit of evaluation. Four choices are
costed for each: the forecast's, hindsight's, the worst available, and running
immediately. All four are costed against realised intensity, because what a
choice was expected to cost is not what it cost.

Three decisions in the design. Regret is normalised by the available spread,
since an unnormalised figure rewards a scheduler for calm weather rather than
for judgement. Decisions whose window is flat are excluded and counted rather
than scored as successes, for the same reason. And a decision whose window has
a gap in the settled record is skipped rather than interpolated: a scheduler
scored on invented observations cannot be falsified.

Baselines are scored twice: once on the decisions the captured forecasts faced,
and once across the whole record. The first version reported only the second and
set it beside the published forecast, which is not a comparison — the captured
decisions come from a few weeks of one season, and on a six-hour window the
difference in available spread was large enough to reverse the ranking and make
a weekly average appear to beat a production forecast. Aligning the samples
required aligning the windows as well as the issue times, since a captured
forecast is issued partway through a period and its window begins at the next
one; matching on the issue minute alone shifted the baseline half an hour.

An earlier claim in this project's planning was wrong and is corrected here.
Regret against the published forecast is not computable over the historical
record, because a decision needs the forecast as it stood at the issue moment
and the only source of that is the captured forward snapshots. The revised
forecast field in a backfilled range response cannot serve, for the same reason
it cannot serve as a benchmark. Baselines are unaffected, since they can be
evaluated at any past moment, so the two are reported on separate samples.

## Auditing the decision comparison

The baselines capture more of the available saving than the published forecast
on identical decisions, which is surprising enough to need opening up rather
than reporting.

The first hypothesis was that the baseline was seeing something it should not.
It was not, and the point is structural rather than empirical: a seasonal
reference sits seven days before its target, while an issue time sits at most
forty-eight hours before it, so the reference has settled at least 118.5 hours
earlier in the tightest case. There is no configuration in which the
availability gate binds. A test records the margin.

What the audit shows instead is a difference in the shape of the error. Over a
day-long window the published forecast lands within one period of the optimum on
half of decisions against the baseline's eighth, and lands more than twelve
periods away on a fifth against the baseline's eighth. It is right far more
often and wrong more expensively. The costliest decisions are ones where it
forecast an overnight period 30 to 43 gCO2/kWh cleaner than it turned out, and
the scheduler committed to it; the baseline, having no opinion beyond the weekly
average, stayed near the optimum. A mean regret conflates those two failure
modes, and the distance-from-optimum breakdown separates them.

One statistic had to be corrected. Window flatness was defined as the gap
between the best placement and the tenth-best, which is a modest position among
the forty-five a day-long window offers and the second-worst among the eleven in
six hours. The figure therefore described the window length rather than the
shape of the grid, and made the two window sizes look incomparable when they
were not. It is now taken at rank fractions.

## Revision dynamics

The captured forecast vintages support a question the source cannot answer about
itself: whether the published forecast revises efficiently. Under efficiency
successive revisions are serially uncorrelated, because each incorporates
everything known at the time and nothing about what comes next. Positive
autocorrelation indicates under-reaction, and implies part of the coming
revision is predictable from the last.

The captured series is not the forecaster's revision sequence. About half of
captures find the value unchanged, so the differenced captured series is padded
with zeros; consecutive identical values are collapsed before any inference.
Collapsing also yields a quantity the API does not expose — how often the
forecast is actually revised, by lead time.

One hypothesis was raised and discarded during this work. The captured
autocorrelation of roughly minus a half was first attributed to those repeats. A
test showed the opposite: zeros dilute a correlation towards nothing. Exactly
minus a half is the signature of a stable level observed with independent noise,
where consecutive differences share a term with opposite sign — which implies a
forecast that retraces its own movement rather than converging. The path summary
corroborates it independently: median total movement of 117.5 gCO2/kWh against
median net movement of 12, so most movement is undone. Both mechanisms now have
their own test so the distinction cannot be lost later.

Two measurement points. Accuracy against lead time is computed on periods
forecast at every lead, since pooling all rows compares different days at
different horizons and produces a non-monotonic curve that reflects weather
rather than horizon — an error this project made in an earlier lead-time table.
And a correlation over revisions of identical size is undefined rather than
zero; reporting zero there would claim efficiency on the strength of a
degenerate sample, so the undefined case is left undefined and tested for.

## Correcting the forecast

The revision analysis implies a correction rather than merely describing a
property. If revisions are anticorrelated and a revision predicts the error
remaining with a negative sign, the forecast overshoots, and subtracting part of
its latest revision should reduce error. The coefficient is the least-squares
slope of the remaining error on the revision, available in closed form, so there
is no search and nothing tuned by hand.

Three constraints make the result meaningful. Fitting is per lead band, because
the twelve-hour boundary separates two regimes with typical revision sizes an
order of magnitude apart. The split is by date, since revisions of the same
target are not independent and a row-wise split would put a period's early
revisions in training and its later ones in test. And a band with fewer than
thirty observations is left uncorrected: passing the published forecast through
unchanged is the safe default, where a coefficient fitted on a handful of rows
is noise given a decision to make.

A test fixture caught something about the storage layout in passing. Writing one
period per snapshot puts several files at the same capture minute, and the later
ones overwrite the earlier. Real captures carry every period in a single payload,
so the production path is unaffected, but a fixture that does not imitate that
shape silently loses most of its own data.

## Delivered capture rate

The workflow asks for a run every hour. It receives fewer, and the shortfall is
a property of the platform rather than of this repository.

Observed, against a nominal half-hourly schedule: 38 runs on 23 August, 27 on
the 24th, 26 on the 25th, 17 on the 26th, then 2 on the 27th and 1 on the 28th.
The runs that started all succeeded in 25 to 28 seconds. One failure in that
period, run #178, was a push rejected because a human push landed first, and the
commit step now rebases and retries.

Two candidate causes were checked and dropped. The repository is 2.38 MiB
packed, so size is not the constraint — git deltas these repetitive payloads
almost perfectly. And the runs are not failing, they are not being started, so
nothing in the code path is responsible.

The cadence was reduced to hourly on that evidence, on the hypothesis that a
lower nominal rate is deprioritised less aggressively. It delivered the same two
runs a day, so the hypothesis was wrong and the ask was not the constraint. Two
a day is what free scheduled Actions gives this repository.

The consequence is stated rather than worked around. Moving capture to a machine
under our own control would restore the rate, but the extra data would be
another week of late summer, and the open question — whether the correction's
coefficient survives a winter — needs months rather than days. The revision and
correction analyses rest on 20-26 August, when capture was dense, and the
repository says so.

The design already assumes this. Each run re-harvests the past 24 hours, so a
missed run costs forecast vintages but no outcomes. The redundancy was added on
the second day for exactly this reason, before the delivery rate was known.

## Intervals on the correction

An improvement without an interval cannot be read. The correction reduces error
by 6.7% at 6-12 hours and by 0.3% at 0-3, and only one of those is worth
believing; the point estimates alone do not say which.

The interval comes from a paired bootstrap. Paired because both forecasts are
scored on the same resampled rows, so the interval describes the difference
rather than the variability of either. Resampled by target period rather than by
row, because several revisions of one period appear in the frame and their
errors move together — treating them as independent would give an interval far
too narrow, and the same reasoning governs the train-test split.

## Splitting by date, and where that goes wrong

The train-test split takes the first 60% of dates. The reasoning is sound —
revisions of the same target period are not independent, so a row-wise split
would put a period's early revisions in training and its later ones in test —
but the implementation has a fault that only appeared once the capture rate
changed.

A fixed fraction of dates is not a fixed fraction of rows. Capture fell from
around 38 runs a day to 2, so the later dates carry a small share of the data.
On 28 August the split gave 5,261 training revisions against 3,509 test; on the
29th, with three more sparse days added, it gave 8,265 against 797. The test
half had shrunk by a factor of four while appearing, by the date count, to have
grown.

The consequence is visible in the gate's first run. Bands that were significant
the day before, with improvements of +0.74 and +0.99 gCO2/kWh, reported +0.19
and -1.19 with intervals spanning zero. That is not the correction failing: the
coefficients barely moved, 0.44, 0.48 and 0.47 against 0.44, 0.47 and 0.46 the
day before. It is a held-out sample too thin to detect anything, and the
intervals correctly declining to claim otherwise.

Both consequences were acted on. The split now chooses the date boundary that
puts the requested share of *rows* on the training side, rather than the date at
a fixed position, so a changing capture rate cannot silently starve one half.
The boundary is still a date, so no target period straddles the split and the
independence argument is unaffected. Days are indivisible, so the requested
fraction is rarely exactly attainable and the nearest achievable boundary is
taken.

And the gate now keeps every run's fitted coefficient, promoted or not, so that
stability over time is visible. A coefficient settling near one value across
refits is evidence no single run can give, since one run's interval speaks only
to that run's sample. A coefficient wandering is the opposite, and it will not
always trip the drift check, which compares consecutive runs rather than the
trend. Coefficients from bands that failed are kept as well: whether a failing
band's coefficient is stable is what later distinguishes too little data from no
effect at all.

## The promotion gate

The correction was significant once, on four days of one week. The gate exists
to keep asking whether it still is.

Three conditions, and the third is the one worth defending. An interval
excluding zero says the improvement is unlikely to be chance on this sample; it
says nothing about whether the coefficient is a property of the forecast. A
refit that moves the coefficient from 0.46 to 0.93 is describing the sample,
and the gate rejects it on stability grounds even where its interval is clean.

The build fails only on regression — a band that had been promoted and no longer
is. Failing whenever any band falls short would leave the build red from the
first run, since most bands never clear the bar, and a permanently red gate is
one nobody reads. A promoted band that vanishes from the evaluation entirely
counts as regressing: its recorded coefficient is then standing on nothing, and
the absence of a row reporting it makes the situation easy to miss.

On a pull request the gate reports but does not write the record. A branch must
not be able to promote a coefficient into the record it is being judged against.

## Known limitations

Scheduled workflows on GitHub are best-effort. Runs are delayed under load and
occasionally dropped, so the capture record will contain gaps. Gaps are recorded
rather than interpolated, and any evaluation must be conditioned on the periods
actually captured rather than assuming a complete grid.

GitHub also disables scheduled workflows in repositories that go inactive for an
extended period. Capture commits may not count as activity for that purpose, so
the workflow state needs occasional checking.

The historical `forecast` values recovered by backfill are revised values, not
values as issued. They are stored because they are useful for characterising the
published forecast in aggregate, but they must not be used as features or as a
benchmark at a stated lead time.

Regional data are forecast-only: NESO publishes no realised regional intensity.
Supervised work is therefore confined to the national series.

## Assembling the record

The loader produces three things. A forecast record of one row per period and
issue time, built only from forward snapshots. An outcome record of one row per
period, taking the observation with the latest capture time as final, since the
API settles a value some minutes after a period ends and may revise it. An
evaluation frame joining the two, with error signed as forecast minus outcome.

Three decisions there are worth stating. Forecasts at non-positive lead time are
dropped: the first period of a forward response is the one already in progress
and is an observation, not a prediction. The forecast record carries no outcome
column, so the outcome can only arrive through the join. And the exclusion of
backfilled forecast values is enforced by a test rather than left to
convention, because the failure it prevents is a leak that would improve every
metric while invalidating all of them.

Errors are reported bucketed by lead time. A single average across all horizons
conflates a half-hour-ahead problem with a forty-eight-hour-ahead one.

## Next

Feature construction from the captured record, a seasonal baseline, a residual
correction model benchmarked against the published forecast at matched lead
times, a promotion gate that fails the build when the candidate does not beat
both, and drift monitoring on inputs and residuals.

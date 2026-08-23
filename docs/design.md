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

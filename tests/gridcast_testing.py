"""Helpers shared between test modules.

A plain module rather than an import from another test file: the tests
directory is not a package, so ``from tests.test_models import ...`` depends on
how pytest was invoked and fails outside the project root. pytest adds each test
directory to the path, so a sibling module is importable by name wherever the
suite is run from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gridcast.models import DEFAULT_PARAMS

#: Fewer boosting iterations than the default, so tests stay quick. The
#: structure being checked is learnable well inside this budget.
FAST = {**DEFAULT_PARAMS, "max_iter": 40}


def synthetic_dataset(days: int = 120, window_periods: int = 12, seed: int = 0):
    """Decisions whose windows have a learnable shape plus noise.

    Built directly rather than through a store so the tests stay fast and the
    signal is known: intensity follows the position within the window, which a
    model can learn from the position feature alone.
    """
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2026-01-01T18:00Z")

    for day in range(days):
        issue = start + pd.Timedelta(days=day)
        level = 200.0 + rng.normal(0, 30)
        for position in range(window_periods):
            shape = 40.0 * np.sin(2 * np.pi * position / window_periods)
            rows.append(
                {
                    "decision_id": issue,
                    "captured_at": issue,
                    "period_start": issue + pd.Timedelta(minutes=30 * position),
                    "position": position,
                    "actual": level + shape + rng.normal(0, 3),
                    "horizon_hours": position * 0.5,
                    "sin_day": np.sin(2 * np.pi * position / window_periods),
                    "cos_day": np.cos(2 * np.pi * position / window_periods),
                    "date": issue.date(),
                }
            )

    frame = pd.DataFrame(rows)
    frame["actual_rel"] = frame["actual"] - frame.groupby("decision_id")["actual"].transform("mean")
    frame["sin_day_rel"] = frame["sin_day"] - frame.groupby("decision_id")["sin_day"].transform(
        "mean"
    )

    outcomes = frame[["period_start", "actual"]].drop_duplicates("period_start")
    return frame, outcomes.reset_index(drop=True)

"""Evaluation against externally-defined market stress dates.

There are no ground-truth anomaly labels, so this is a weak evaluation, and it
is worth being precise about what it can and cannot show.

`STRESS_DATES` is a fixed list of US equity sessions in the test window on which
the S&P 500 moved more than roughly 4.5% in either direction, including the four
days that triggered the NYSE level-1 circuit breaker. It comes from index
history, not from this pipeline's own features, and it was written down before
any detector output was inspected. Nothing in the pipeline is tuned against it.

What recall on this list shows: the detector fires on days that were
unambiguously abnormal. What it does not show: precision. A detector that flags
every day scores perfect recall, so recall is only meaningful read next to the
flag rate, and both are reported together below.
"""

from __future__ import annotations

import pandas as pd

STRESS_DATES = [
    "2020-02-24",  # -3.4% S&P, first COVID gap down
    "2020-02-25",  # -3.0%
    "2020-02-27",  # -4.4%
    "2020-03-09",  # -7.6%, circuit breaker
    "2020-03-11",  # -4.9%, bear market confirmed
    "2020-03-12",  # -9.5%, circuit breaker
    "2020-03-13",  # +9.3%
    "2020-03-16",  # -12.0%, circuit breaker
    "2020-03-17",  # +6.0%
    "2020-03-18",  # -5.2%, circuit breaker
    "2020-03-24",  # +9.4%
    "2020-03-26",  # +6.2%
]


def evaluate_detector(
    df: pd.DataFrame, flag_col: str, split: str = "test"
) -> dict[str, float]:
    """Recall on the stress list and overall flag rate for one detector column."""
    scoped = df[df["split"] == split]
    if scoped.empty:
        return {"recall": float("nan"), "flag_rate": float("nan"), "n_events": 0}

    fired_dates = set(
        pd.to_datetime(scoped.loc[scoped[flag_col] == 1, "Date"]).dt.normalize()
    )
    available = set(pd.to_datetime(scoped["Date"]).dt.normalize())
    events = [d for d in pd.to_datetime(STRESS_DATES) if d in available]

    hits = sum(1 for d in events if d in fired_dates)
    return {
        "recall": hits / len(events) if events else float("nan"),
        "flag_rate": float(scoped[flag_col].mean()),
        "n_events": len(events),
        "n_hits": hits,
    }


def comparison_table(df: pd.DataFrame, flag_cols: list[str]) -> pd.DataFrame:
    """One row per detector: how often it fires, and how much of the list it catches."""
    rows = []
    for col in flag_cols:
        result = evaluate_detector(df, col)
        rows.append(
            {
                "detector": col,
                "flag_rate": result["flag_rate"],
                "stress_days_caught": f"{result['n_hits']}/{result['n_events']}",
                "recall": result["recall"],
            }
        )
    return pd.DataFrame(rows)

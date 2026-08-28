"""Cross-sectional market layer.

The original notebook computed the "extreme move" cut as

    market_stats['market_ret'].abs().quantile(0.95)

over the whole series. That quantile is a function of March 2020, so the test
period was helping to define what counted as extreme in the test period. Here
the cut is estimated on the training split and then applied unchanged forward,
which is the same discipline the scaler and the K-Means threshold follow.
"""

from __future__ import annotations

import pandas as pd

from . import config

BREADTH_FLOOR = 0.30   # fixed prior, not fitted: fewer than 30% of names green
MARKET_PCTILE = 0.95   # quantile of |market return|, estimated on train only


def build_market_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-ticker rows into one row per date."""
    stats = (
        df.groupby("Date")
        .agg(
            market_ret=("ret", "mean"),
            breadth=("ret", lambda s: float((s > 0).mean())),
            flag_rate=("final_anomaly", "mean"),
            n_names=("ret", "size"),
        )
        .reset_index()
    )

    split_by_date = df.groupby("Date")["split"].first()
    stats["split"] = stats["Date"].map(split_by_date)

    train_ret = stats.loc[stats["split"] == "train", "market_ret"].abs()
    if train_ret.empty:
        raise ValueError("no training dates available to fit the market cut")
    threshold = float(train_ret.quantile(MARKET_PCTILE))

    stats["market_move_extreme"] = (stats["market_ret"].abs() > threshold).astype(int)
    stats["breadth_collapse"] = (stats["breadth"] < BREADTH_FLOOR).astype(int)
    stats["mkt_anomaly"] = (
        stats["breadth_collapse"] | stats["market_move_extreme"]
    ).astype(int)

    stats.attrs["market_threshold"] = threshold
    return stats

"""Rolling feature construction.

Every feature at date *t* is a function of data strictly at or before *t*, and
the *distribution* each observation is scored against uses only dates strictly
before *t*. Concretely:

    ret_z[t]      = (ret[t] - mean(ret[t-63 : t-1])) / std(ret[t-63 : t-1])
    volz[t]       = (log_vol[t] - mean(log_vol[t-21 : t-1])) / std(...)
    range_pct[t]  = fraction of raw_range[t-63 : t-1] below raw_range[t]

The `.shift(1)` before each rolling window is what excludes the current day from
its own reference distribution. Without it, a large move inflates the standard
deviation it is being divided by and the z-score is biased toward zero — the
feature quietly launders away the very events the pipeline is meant to find.

`tests/test_features.py` asserts this causality directly: appending future rows
must not change any already-computed value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _percentile_rank_of_last(window: np.ndarray) -> float:
    """Fraction of the trailing window strictly below today's value."""
    return float((window[:-1] < window[-1]).mean())


def compute_ticker_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling features for a single ticker's price history."""
    df = df.sort_values("Date").copy()

    # A. Simple close-to-close return.
    df["ret"] = df["Close"].pct_change()

    # B. Return z-score against the trailing quarter, current day excluded.
    trailing_ret = df["ret"].shift(1).rolling(config.W_RET)
    sigma_ret = trailing_ret.std().replace(0, np.nan)
    df["ret_z"] = (df["ret"] - trailing_ret.mean()) / sigma_ret

    # C. Log-volume z-score. Logs first: raw volume is right-skewed enough that
    #    a z-score on it is dominated by a handful of heavy days.
    df["log_vol"] = np.log(df["Volume"].replace(0, np.nan))
    trailing_vol = df["log_vol"].shift(1).rolling(config.W_VOL)
    sigma_vol = trailing_vol.std().replace(0, np.nan)
    df["volz"] = (df["log_vol"] - trailing_vol.mean()) / sigma_vol

    # D. Intraday range as a percentile rank of its own trailing history.
    #    Rank rather than z-score because the range distribution is bounded
    #    below by zero and badly non-normal.
    df["raw_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["range_pct"] = (
        df["raw_range"]
        .rolling(window=config.W_RANGE + 1)
        .apply(_percentile_rank_of_last, raw=True)
    )

    return df


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply `compute_ticker_features` to every ticker and drop warm-up rows."""
    frames = [
        compute_ticker_features(block)
        for _, block in panel.groupby("Ticker", sort=True)
    ]
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=config.RAW_FEATURES)
    return out.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def add_split_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each row as train / val / test / excluded from the config dates."""
    df = df.copy()
    date = pd.to_datetime(df["Date"])
    split = pd.Series("excluded", index=df.index, dtype=object)
    for name, (lo, hi) in (
        ("train", config.TRAIN),
        ("val", config.VAL),
        ("test", config.TEST),
    ):
        split[(date >= lo) & (date <= hi)] = name
    df["split"] = split
    return df

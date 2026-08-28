"""Synthetic OHLCV panel with a planted stress regime.

This exists so the test suite and `--synthetic` smoke runs work offline, and so
the block-size behaviour of DBSCAN can be demonstrated on data whose regime
shift is known by construction. It is *not* a stand-in for the real results: the
numbers in the README come from the cached Yahoo panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

STRESS_START = "2020-02-24"
STRESS_END = "2020-03-31"


def make_synthetic_panel(
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Correlated GBM with a sustained high-volatility, negative-drift regime."""
    tickers = tickers or config.TICKERS
    dates = pd.bdate_range(
        start or config.DOWNLOAD_START, end or config.DOWNLOAD_END
    )
    rng = np.random.default_rng(seed)

    stress = (dates >= STRESS_START) & (dates <= STRESS_END)
    daily_vol = np.where(stress, 0.055, 0.011)
    daily_drift = np.where(stress, -0.010, 0.0006)

    # A shared market factor is what makes the stress regime *dense* in feature
    # space: during the crash every name moves together, day after day.
    market = rng.standard_normal(len(dates)) * daily_vol + daily_drift

    frames = []
    for i, ticker in enumerate(tickers):
        beta = 0.8 + 0.1 * i
        idio = rng.standard_normal(len(dates)) * daily_vol * 0.45
        ret = beta * market + idio

        close = 100.0 * np.exp(np.cumsum(ret))
        span = np.abs(rng.standard_normal(len(dates))) * daily_vol * 1.3
        high = close * (1 + span)
        low = close * (1 - span)
        open_ = close * (1 + rng.standard_normal(len(dates)) * daily_vol * 0.3)

        base_vol = 5e6 * (1 + 0.3 * i)
        volume = base_vol * np.exp(
            rng.standard_normal(len(dates)) * 0.25 + np.where(stress, 0.9, 0.0)
        )

        frames.append(
            pd.DataFrame(
                {
                    "Date": dates,
                    "Ticker": ticker,
                    "Open": open_,
                    "High": np.maximum.reduce([high, close, open_]),
                    "Low": np.minimum.reduce([low, close, open_]),
                    "Close": close,
                    "Volume": volume.round(),
                }
            )
        )

    return pd.concat(frames, ignore_index=True).sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

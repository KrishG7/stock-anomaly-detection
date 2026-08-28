"""Price ingestion.

Yahoo data is not stable over time: tickers get adjusted, the endpoint changes,
and `yfinance` breaks periodically. The first successful download is therefore
written to `data/prices.parquet` and committed, so that every later run of this
repo reproduces the exact numbers in the README rather than whatever Yahoo
happens to serve that day.
"""

from __future__ import annotations

import pandas as pd

from . import config

COLUMNS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]


def download_prices(
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch a long-format OHLCV panel from Yahoo. Requires network access."""
    import yfinance as yf  # imported lazily so cached runs need no network

    tickers = tickers or config.TICKERS
    start = start or config.DOWNLOAD_START
    end = end or config.DOWNLOAD_END

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )

    frames = []
    for ticker in tickers:
        if ticker not in raw.columns.get_level_values(0):
            raise KeyError(f"Yahoo returned no data for {ticker}")
        block = raw[ticker].copy()
        block["Ticker"] = ticker
        frames.append(block.reset_index())

    panel = pd.concat(frames, ignore_index=True)
    panel = panel[COLUMNS].dropna(subset=["Close", "Volume"])
    return panel.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def load_prices(refresh: bool = False) -> pd.DataFrame:
    """Return the cached panel, downloading and caching it if absent."""
    if config.PRICE_CACHE.exists() and not refresh:
        panel = pd.read_parquet(config.PRICE_CACHE)
        return panel.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    panel = download_prices()
    panel.to_parquet(config.PRICE_CACHE, index=False)
    print(f"Cached {len(panel):,} rows to {config.PRICE_CACHE}")
    return panel


if __name__ == "__main__":
    df = load_prices(refresh=True)
    print(df.groupby("Ticker")["Date"].agg(["min", "max", "count"]))

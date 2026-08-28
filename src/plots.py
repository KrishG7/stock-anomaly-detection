"""Figures. Saved to `figures/`, not shown, so the pipeline runs headless."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config


def plot_flags_over_price(df: pd.DataFrame, split: str = "test") -> None:
    """Close price per ticker with consensus-flagged sessions marked."""
    scoped = df[df["split"] == split].copy()
    tickers = sorted(scoped["Ticker"].unique())

    fig, axes = plt.subplots(
        len(tickers), 1, figsize=(11, 2.1 * len(tickers)), sharex=True
    )
    axes = [axes] if len(tickers) == 1 else list(axes)

    for ax, ticker in zip(axes, tickers):
        block = scoped[scoped["Ticker"] == ticker].sort_values("Date")
        ax.plot(block["Date"], block["Close"], linewidth=1.1, color="#333333")
        hits = block[block["final_anomaly"] == 1]
        ax.scatter(
            hits["Date"], hits["Close"], s=26, color="#c0392b", zorder=3, label="flagged"
        )
        ax.set_ylabel(ticker, fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.5)

    axes[0].set_title(f"Consensus anomalies over close price — {split} split")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / f"flags_over_price_{split}.png", dpi=140)
    plt.close(fig)


def plot_detector_agreement(df: pd.DataFrame, split: str = "test") -> None:
    """Daily flag rate per detector layer, to show where they diverge."""
    scoped = df[df["split"] == split]
    daily = (
        scoped.groupby("Date")[
            ["rule_anomaly", "kmeans_anomaly", "dbscan_anomaly"]
        ]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(11, 3.6))
    for col, colour in (
        ("rule_anomaly", "#2c3e50"),
        ("kmeans_anomaly", "#2980b9"),
        ("dbscan_anomaly", "#c0392b"),
    ):
        ax.plot(daily["Date"], daily[col], label=col.replace("_anomaly", ""),
                linewidth=1.3, color=colour)

    ax.set_ylabel("fraction of names flagged")
    ax.set_title(f"Detector agreement — {split} split")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / f"detector_agreement_{split}.png", dpi=140)
    plt.close(fig)


def plot_market_layer(stats: pd.DataFrame, split: str = "test") -> None:
    """Market return and breadth, with market-level flags shaded."""
    scoped = stats[stats["split"] == split].sort_values("Date")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    ax1.bar(scoped["Date"], scoped["market_ret"], width=1.0, color="#34495e")
    ax1.set_ylabel("mean return")
    ax1.grid(alpha=0.25, linewidth=0.5)

    ax2.plot(scoped["Date"], scoped["breadth"], color="#16a085", linewidth=1.2)
    ax2.axhline(0.30, linestyle="--", linewidth=0.9, color="#7f8c8d")
    ax2.set_ylabel("breadth")
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.25, linewidth=0.5)

    for _, row in scoped[scoped["mkt_anomaly"] == 1].iterrows():
        for ax in (ax1, ax2):
            ax.axvspan(row["Date"], row["Date"], color="#c0392b", alpha=0.30, linewidth=3)

    ax1.set_title(f"Market layer — {split} split (shaded = market anomaly)")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / f"market_layer_{split}.png", dpi=140)
    plt.close(fig)

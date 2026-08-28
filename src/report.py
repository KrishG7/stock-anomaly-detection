"""Human-readable outputs.

Every reported z-score is the *unscaled* one, so a printed `ret_z` of 3.1 really
does mean 3.1 trailing standard deviations.
"""

from __future__ import annotations

import pandas as pd

from . import config

CARD_COLUMNS = [
    "Date", "Ticker", "rule_label", "ret", "ret_z", "volz", "range_pct", "why",
]


def _why(row: pd.Series) -> str:
    reasons = []
    if row["rule_label"] != "normal":
        reasons.append(f"rule: {row['rule_label']}")
    if row["kmeans_anomaly"] == 1:
        reasons.append("k-means outlier")
    if row["dbscan_anomaly"] == 1:
        reasons.append("dbscan noise")
    return "; ".join(reasons)


def build_anomaly_card(df: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Every flagged (date, ticker) in a split, with its reasons."""
    hits = df[(df["split"] == split) & (df["final_anomaly"] == 1)].copy()
    if hits.empty:
        return pd.DataFrame(columns=CARD_COLUMNS)
    hits["why"] = hits.apply(_why, axis=1)
    card = hits[CARD_COLUMNS].sort_values(["Date", "Ticker"])
    card.to_csv(config.OUTPUT_DIR / f"anomaly_card_{split}.csv", index=False)
    return card


def monthly_report(df: pd.DataFrame, stats: pd.DataFrame, month: str) -> pd.DataFrame:
    """Flagged rows for one calendar month, joined to the market flag."""
    period = pd.to_datetime(df["Date"]).dt.to_period("M")
    hits = df[(period == pd.Period(month)) & (df["final_anomaly"] == 1)].copy()
    if hits.empty:
        print(f"No anomalies in {month}")
        return pd.DataFrame()

    hits = hits.merge(stats[["Date", "mkt_anomaly"]], on="Date", how="left")
    hits["why"] = hits.apply(_why, axis=1)

    report = pd.DataFrame(
        {
            "date": pd.to_datetime(hits["Date"]).dt.date,
            "ticker": hits["Ticker"],
            "type": hits["rule_label"],
            "ret": hits["ret"].round(4),
            "ret_z": hits["ret_z"].round(2),
            "volz": hits["volz"].round(2),
            "mkt_flag": hits["mkt_anomaly"],
            "why": hits["why"],
        }
    ).sort_values(["date", "ticker"])

    report.to_csv(config.OUTPUT_DIR / f"monthly_report_{month}.csv", index=False)
    print(f"\n--- MONTHLY REPORT: {month} ---")
    print(report.to_string(index=False))
    return report


def date_query(df: pd.DataFrame, stats: pd.DataFrame, date: str) -> None:
    """Print the market state and every flagged name for a single session."""
    target = pd.to_datetime(date)
    row = stats[stats["Date"] == target]
    if row.empty:
        print(f"No data for {date} (market holiday, or outside the loaded range).")
        return

    state = "STRESSED" if row["mkt_anomaly"].iloc[0] else "normal"
    print(f"\n=== {date} — market: {state} ===")
    print(
        f"    mean return {row['market_ret'].iloc[0]:+.2%}   "
        f"breadth {row['breadth'].iloc[0]:.2f}"
    )

    hits = df[(df["Date"] == target) & (df["final_anomaly"] == 1)]
    if hits.empty:
        print("    no individual names flagged")
        return

    view = hits[
        ["Ticker", "rule_label", "ret", "ret_z", "volz",
         "rule_anomaly", "kmeans_anomaly", "dbscan_anomaly"]
    ].copy()
    view[["ret_z", "volz"]] = view[["ret_z", "volz"]].round(2)
    view["ret"] = view["ret"].map(lambda v: f"{v:+.2%}")
    print(view.to_string(index=False))

#!/usr/bin/env python3
"""End-to-end run.

    python run_pipeline.py                # cached Yahoo panel (downloads once)
    python run_pipeline.py --synthetic    # offline smoke run, no network
    python run_pipeline.py --refresh      # re-download and re-cache

Everything downstream of the splits obeys one rule: nothing fitted, calibrated,
or thresholded may see a date later than the split it was fitted on.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src import config, detectors, evaluate, features, market, plots, report


def build(panel: pd.DataFrame, block_freq: str) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    df = features.add_split_labels(features.build_features(panel))

    scaler = detectors.fit_scaler(df)
    df = detectors.apply_scaler(df, scaler)

    df = detectors.rule_labels(df)
    df, _, kmeans_threshold = detectors.kmeans_layer(df)

    eps = detectors.calibrate_eps(df, block_freq=block_freq)
    df["dbscan_anomaly"] = detectors.dbscan_walk_forward(
        df,
        score_mask=df["split"] == "test",
        history_mask=df["split"].isin(["train", "val"]),
        eps=eps,
        block_freq=block_freq,
    )

    df["final_anomaly"] = df[
        ["rule_anomaly", "kmeans_anomaly", "dbscan_anomaly"]
    ].max(axis=1)

    stats = market.build_market_stats(df)
    print(
        f"  k-means distance cut (train p{config.KMEANS_PCTILE}): {kmeans_threshold:.3f}\n"
        f"  market |return| cut (train p95): {stats.attrs['market_threshold']:.4f}"
    )
    return df, stats, eps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="use generated data instead of Yahoo")
    parser.add_argument("--refresh", action="store_true",
                        help="force a fresh download")
    args = parser.parse_args()

    if args.synthetic:
        from src.synthetic import make_synthetic_panel
        panel = make_synthetic_panel()
        print("Using SYNTHETIC data — results are illustrative only.\n")
    else:
        from src.ingest import load_prices
        panel = load_prices(refresh=args.refresh)

    print(f"Loaded {len(panel):,} rows, "
          f"{panel['Ticker'].nunique()} tickers, "
          f"{panel['Date'].min().date()} to {panel['Date'].max().date()}\n")

    # DBSCAN is transductive, so the size of each scoring block changes the
    # answer. Run both and report the gap rather than picking one silently.
    results = {}
    for block_freq, label in (("D", "daily blocks"), ("ME", "monthly blocks")):
        print(f"[{label}]")
        df, stats, eps = build(panel, block_freq)
        results[block_freq] = (df, stats, eps)
        dbscan_test = df.loc[df["split"] == "test", "dbscan_anomaly"]
        print(f"  dbscan flagged {int(dbscan_test.sum())} of "
              f"{len(dbscan_test)} test rows (eps={eps})\n")

    df, stats, _ = results["D"]

    print("=== Test-split evaluation (daily blocks) ===")
    table = evaluate.comparison_table(
        df, ["rule_anomaly", "kmeans_anomaly", "dbscan_anomaly", "final_anomaly"]
    )
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    monthly_dbscan = results["ME"][0]
    print("\n=== DBSCAN block-size sensitivity ===")
    for freq, name in (("D", "daily"), ("ME", "monthly")):
        d = results[freq][0]
        res = evaluate.evaluate_detector(d, "dbscan_anomaly")
        print(f"  {name:<8} flag rate {res['flag_rate']:.3f}   "
              f"stress days caught {res['n_hits']}/{res['n_events']}")

    card = report.build_anomaly_card(df)
    print(f"\nWrote {len(card)} flagged rows to outputs/anomaly_card_test.csv")

    for month in ("2020-01", "2020-02", "2020-03"):
        report.monthly_report(df, stats, month)

    report.date_query(df, stats, "2020-03-16")

    plots.plot_flags_over_price(df)
    plots.plot_detector_agreement(df)
    plots.plot_market_layer(stats)
    print(f"\nFigures written to {config.FIGURE_DIR}")


if __name__ == "__main__":
    main()

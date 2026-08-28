"""Detector layers: rule-based, K-Means distance, and walk-forward DBSCAN.

Two design points worth stating up front.

1. Scaling is written to *new* columns with an `_s` suffix. The rule layer and
   every report read the unscaled z-scores, so "|ret_z| > 2.5" still means two
   and a half standard deviations of the trailing return distribution. Scaling
   in place would silently redefine that threshold in units of the 2018
   dispersion of the z-score itself.

2. DBSCAN has no `predict`. Scoring new data means refitting on history plus the
   new rows, which makes the result depend on how large a block you hand it.
   `block_freq` exposes that choice instead of hiding it — see the README.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

from . import config


# --------------------------------------------------------------------------
# Scaling
# --------------------------------------------------------------------------
def fit_scaler(df: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on the training split only."""
    train = df.loc[df["split"] == "train", config.RAW_FEATURES]
    if train.empty:
        raise ValueError("no training rows to fit the scaler on")
    return StandardScaler().fit(train)


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Write scaled copies of the features, leaving the originals untouched."""
    df = df.copy()
    df[config.SCALED_FEATURES] = scaler.transform(df[config.RAW_FEATURES])
    return df


# --------------------------------------------------------------------------
# Layer 1 — rule based
# --------------------------------------------------------------------------
def rule_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised threshold rules on the unscaled z-scores."""
    df = df.copy()
    extreme = df["ret_z"].abs() > config.RULE_Z
    crash = extreme & (df["ret"] < 0)
    spike = extreme & (df["ret"] > 0)
    vol_shock = df["volz"] > config.RULE_VOLZ

    parts = pd.DataFrame(
        {
            "crash": np.where(crash, "crash", ""),
            "spike": np.where(spike, "spike", ""),
            "volume_shock": np.where(vol_shock, "volume_shock", ""),
        },
        index=df.index,
    )
    joined = parts.apply(lambda r: " + ".join(x for x in r if x), axis=1)

    df["rule_label"] = joined.replace("", "normal")
    df["rule_anomaly"] = (df["rule_label"] != "normal").astype(int)
    return df


# --------------------------------------------------------------------------
# Layer 2 — K-Means distance to nearest centroid
# --------------------------------------------------------------------------
def kmeans_layer(df: pd.DataFrame) -> tuple[pd.DataFrame, KMeans, float]:
    """Flag rows far from every centroid fitted on the training split."""
    df = df.copy()
    train = df.loc[df["split"] == "train", config.SCALED_FEATURES]

    model = KMeans(
        n_clusters=config.KMEANS_K,
        random_state=config.RANDOM_STATE,
        n_init=10,
    ).fit(train)

    df["kmeans_dist"] = model.transform(df[config.SCALED_FEATURES]).min(axis=1)
    threshold = float(
        np.percentile(
            df.loc[df["split"] == "train", "kmeans_dist"], config.KMEANS_PCTILE
        )
    )
    df["kmeans_anomaly"] = (df["kmeans_dist"] > threshold).astype(int)
    return df, model, threshold


# --------------------------------------------------------------------------
# Layer 3 — walk-forward DBSCAN
# --------------------------------------------------------------------------
def dbscan_walk_forward(
    df: pd.DataFrame,
    score_mask: pd.Series,
    history_mask: pd.Series,
    eps: float,
    block_freq: str = "D",
    min_samples: int = config.DBSCAN_MIN_SAMPLES,
) -> pd.Series:
    """Score `score_mask` rows in chronological blocks, growing the history.

    `block_freq` is the size of each scoring block: "D" for one trading day at a
    time, "ME" for a calendar month. Larger blocks let a sustained regime shift
    supply its own neighbours and be absorbed as a cluster rather than flagged
    as noise. This is a property of DBSCAN's density definition, not a bug.
    """
    flags = pd.Series(0, index=df.index, dtype=int)

    history = df.loc[history_mask, config.SCALED_FEATURES]
    scored = df.loc[score_mask]
    if scored.empty:
        return flags

    period = pd.to_datetime(scored["Date"]).dt.to_period(
        "D" if block_freq == "D" else "M"
    )

    for _, block_idx in scored.groupby(period, sort=True).groups.items():
        block = df.loc[block_idx, config.SCALED_FEATURES]
        combined = pd.concat([history, block])

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(combined)
        flags.loc[block_idx] = (labels[-len(block) :] == -1).astype(int)

        history = combined  # the block joins the history for the next step

    return flags


def calibrate_eps(
    df: pd.DataFrame,
    eps_grid: list[float] | None = None,
    block_freq: str = "D",
    budget: float = config.ALARM_BUDGET,
) -> float:
    """Pick the eps whose validation-split flag rate is closest to `budget`.

    No labels are used. The validation split only fixes how often the detector
    is allowed to fire, which is what makes DBSCAN comparable to the
    97.5th-percentile K-Means rule instead of an arbitrary knob.
    """
    eps_grid = eps_grid or config.DBSCAN_EPS_GRID
    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"

    results = []
    for eps in eps_grid:
        flags = dbscan_walk_forward(
            df, val_mask, train_mask, eps=eps, block_freq=block_freq
        )
        rate = float(flags[val_mask].mean())
        results.append((eps, rate))

    best = min(results, key=lambda t: abs(t[1] - budget))
    print("  eps calibration on validation split:")
    for eps, rate in results:
        marker = "  <-- selected" if eps == best[0] else ""
        print(f"    eps={eps:<5} flag rate={rate:6.2%}{marker}")
    return best[0]

"""Tests for the causality and fitting claims the README makes.

The interesting ones are `test_features_are_causal` and
`test_scaler_sees_only_training_rows`: they check the properties that make the
"leakage-safe" label true, rather than checking that the code runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, detectors, features, market
from src.synthetic import make_synthetic_panel


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return make_synthetic_panel(seed=7)


@pytest.fixture(scope="module")
def built(panel: pd.DataFrame) -> pd.DataFrame:
    df = features.add_split_labels(features.build_features(panel))
    scaler = detectors.fit_scaler(df)
    df = detectors.apply_scaler(df, scaler)
    df = detectors.rule_labels(df)
    df, _, _ = detectors.kmeans_layer(df)
    df["dbscan_anomaly"] = 0
    df["final_anomaly"] = df[["rule_anomaly", "kmeans_anomaly"]].max(axis=1)
    return df


def test_features_are_causal(panel: pd.DataFrame) -> None:
    """Truncating the future must not change any past feature value.

    This is the property the `.shift(1)` calls exist to guarantee. If a rolling
    window ever included the current or a future observation, the values on the
    truncated run would differ.
    """
    full = features.build_features(panel)
    cutoff = pd.Timestamp("2019-06-30")
    truncated = features.build_features(panel[panel["Date"] <= cutoff])

    keys = ["Date", "Ticker"]
    merged = full[full["Date"] <= cutoff].merge(
        truncated, on=keys, suffixes=("_full", "_trunc")
    )
    assert len(merged) > 1000, "truncated overlap is too small to be meaningful"

    for col in config.RAW_FEATURES:
        np.testing.assert_allclose(
            merged[f"{col}_full"], merged[f"{col}_trunc"], rtol=1e-9, atol=1e-12
        )


def test_range_pct_is_a_valid_percentile(built: pd.DataFrame) -> None:
    values = built["range_pct"]
    assert values.between(0.0, 1.0).all()
    # With a 63-day lookback the rank can only take multiples of 1/63.
    assert np.allclose(values * config.W_RANGE, (values * config.W_RANGE).round())


def test_scaler_sees_only_training_rows(built: pd.DataFrame) -> None:
    """The scaler's mean must match the training slice, not the full sample."""
    scaler = detectors.fit_scaler(built)
    train = built.loc[built["split"] == "train", config.RAW_FEATURES]
    np.testing.assert_allclose(scaler.mean_, train.mean().to_numpy(), rtol=1e-9)

    full_mean = built[config.RAW_FEATURES].mean().to_numpy()
    assert not np.allclose(scaler.mean_, full_mean, rtol=1e-3), (
        "train and full-sample means are indistinguishable; "
        "this test cannot detect leakage on this fixture"
    )


def test_raw_z_scores_survive_scaling(built: pd.DataFrame) -> None:
    """Scaling writes to `_s` columns and leaves the interpretable ones alone."""
    assert set(config.SCALED_FEATURES).issubset(built.columns)
    # A trailing z-score has roughly unit dispersion by construction; if the
    # scaler had overwritten it, the standard deviation would be exactly 1.0
    # on the training split.
    train_std = built.loc[built["split"] == "train", "ret_z"].std()
    assert not np.isclose(train_std, 1.0, atol=1e-6)


def test_rule_layer_matches_its_stated_threshold(built: pd.DataFrame) -> None:
    """Every rule flag must be explainable by the documented cut-offs."""
    flagged = built[built["rule_anomaly"] == 1]
    justified = (flagged["ret_z"].abs() > config.RULE_Z) | (
        flagged["volz"] > config.RULE_VOLZ
    )
    assert justified.all()

    unflagged = built[built["rule_anomaly"] == 0]
    assert (unflagged["ret_z"].abs() <= config.RULE_Z).all()
    assert (unflagged["volz"] <= config.RULE_VOLZ).all()


def test_kmeans_threshold_comes_from_training_split(built: pd.DataFrame) -> None:
    train_rate = built.loc[built["split"] == "train", "kmeans_anomaly"].mean()
    expected = 1 - config.KMEANS_PCTILE / 100
    assert abs(train_rate - expected) < 0.01


def test_market_threshold_ignores_test_dates(built: pd.DataFrame) -> None:
    """The market cut must be unchanged if the test split is deleted."""
    with_test = market.build_market_stats(built)
    without_test = market.build_market_stats(built[built["split"] != "test"])
    assert with_test.attrs["market_threshold"] == pytest.approx(
        without_test.attrs["market_threshold"]
    )


def test_dbscan_larger_blocks_flag_no_more_than_daily(built: pd.DataFrame) -> None:
    """Handing over a whole month at once cannot flag *more* than day by day.

    Points in the scoring block count toward each other's neighbour counts, so a
    larger block can only ever supply extra density and absorb borderline points
    into clusters. The effect is directional and, on this fixture, small — see
    the DBSCAN section of the README. It is pinned here so a future change to
    the walk-forward loop cannot reverse it unnoticed.
    """
    test_mask = built["split"] == "test"
    history_mask = built["split"].isin(["train", "val"])

    kwargs = dict(eps=0.6, min_samples=config.DBSCAN_MIN_SAMPLES)
    daily = detectors.dbscan_walk_forward(
        built, test_mask, history_mask, block_freq="D", **kwargs
    )
    monthly = detectors.dbscan_walk_forward(
        built, test_mask, history_mask, block_freq="ME", **kwargs
    )

    assert daily[test_mask].sum() > 0, "eps too tight for this test to mean anything"
    assert monthly[test_mask].sum() <= daily[test_mask].sum()


def test_original_eps_makes_the_layer_inert(built: pd.DataFrame) -> None:
    """Regression guard for the defect this rewrite fixes.

    With eps=2.0 on standardised features almost no point is ever noise, so the
    DBSCAN column is all zeros and the consensus silently reduces to
    rule OR k-means. Calibration exists to prevent exactly this.
    """
    test_mask = built["split"] == "test"
    history_mask = built["split"].isin(["train", "val"])

    inert = detectors.dbscan_walk_forward(
        built, test_mask, history_mask, eps=2.0, block_freq="ME"
    )
    assert inert[test_mask].mean() < 0.02

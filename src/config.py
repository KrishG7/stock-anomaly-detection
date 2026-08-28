"""Central configuration.

Every magic number in the pipeline lives here so that the split boundaries and
window sizes can be audited in one place.
"""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = ROOT / "figures"

for _d in (DATA_DIR, OUTPUT_DIR, FIGURE_DIR):
    _d.mkdir(exist_ok=True)

PRICE_CACHE = DATA_DIR / "prices.parquet"

# --- Universe and date range ----------------------------------------------
TICKERS = ["QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META"]

# Pulled from 2017-09 so that the 63-day rolling windows are fully warm by
# 2018-01-01, the first day of the training split.
DOWNLOAD_START = "2017-09-01"
DOWNLOAD_END = "2020-04-01"

# --- Splits ----------------------------------------------------------------
# Train: fit scaler, K-Means centroids, and every threshold.
# Val:   calibrate DBSCAN eps and the alarm budget. Never used for fitting.
# Test:  held out entirely until the final run.
TRAIN = ("2018-01-01", "2018-12-31")
VAL = ("2019-01-01", "2019-12-31")
TEST = ("2020-01-01", "2020-04-01")

# --- Feature windows -------------------------------------------------------
W_RET = 63    # ~1 quarter, for the return z-score
W_VOL = 21    # ~1 month, for the log-volume z-score
W_RANGE = 63  # lookback for the intraday-range percentile rank

# --- Detector parameters ---------------------------------------------------
RULE_Z = 2.5              # |z| threshold for the rule layer
RULE_VOLZ = 2.5
KMEANS_K = 3
KMEANS_PCTILE = 97.5      # distance percentile on TRAIN that defines "far"
DBSCAN_MIN_SAMPLES = 5
# The original notebook used a single fixed eps=2.0. On standardised features
# that is far too loose: almost nothing is ever labelled noise, so the layer
# contributes nothing to the consensus. The grid is swept downward instead and
# the value chosen by calibration on the validation split.
DBSCAN_EPS_GRID = [0.25, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0]

# Target fraction of days flagged. DBSCAN eps is calibrated on VAL to hit this,
# which is what makes it comparable to the 97.5th-percentile K-Means rule.
ALARM_BUDGET = 0.025

# Feature columns. The `_s` suffix marks the scaled copies used by the
# distance-based models; the unsuffixed columns stay in original z-score units
# so the rule layer and every report remain interpretable.
RAW_FEATURES = ["ret_z", "volz", "range_pct"]
SCALED_FEATURES = [f"{c}_s" for c in RAW_FEATURES]

RANDOM_STATE = 42

"""The same stateless feature mapping is used by training and the API."""
import numpy as np
import pandas as pd

FEATURES = ["log_amount", "log_distance", "new_device", "international",
            "prior_txn_count_1h", "log_prior_mean", "amount_to_prior_mean",
            "hour_sin", "hour_cos"]
INPUTS = ["timestamp", "amount", "distance_km", "new_device", "international",
          "prior_txn_count_1h", "prior_mean_amount"]


def transform(frame: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(frame.timestamp, utc=True)
    hour = ts.dt.hour + ts.dt.minute / 60
    values = {
        "log_amount": np.log1p(frame.amount),
        "log_distance": np.log1p(frame.distance_km),
        "new_device": frame.new_device,
        "international": frame.international,
        "prior_txn_count_1h": frame.prior_txn_count_1h,
        "log_prior_mean": np.log1p(frame.prior_mean_amount),
        "amount_to_prior_mean": (frame.amount / frame.prior_mean_amount.clip(lower=1)).clip(upper=1000),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
    }
    return pd.DataFrame(values, index=frame.index)[FEATURES].astype(float)

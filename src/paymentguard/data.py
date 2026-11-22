"""Synthetic raw events and strictly past-only customer history features."""
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = ["transaction_id", "timestamp", "customer_id", "amount",
               "distance_km", "new_device", "international"]
HISTORY_COLUMNS = ["prior_txn_count_1h", "prior_mean_amount"]


def generate(n: int = 30000, seed: int = 42) -> pd.DataFrame:
    """Generate noisy synthetic labels; these are NOT real fraud observations."""
    if n < 500:
        raise ValueError("Generate at least 500 events.")
    rng = np.random.default_rng(seed)
    seconds = np.sort(rng.choice(n * 120, size=n, replace=False))
    amount = np.round(np.clip(rng.lognormal(3.4, 1.15, n), .01, 20000), 2)
    distance = np.round(rng.exponential(25, n), 2)
    device = rng.binomial(1, .10, n)
    abroad = rng.binomial(1, .12, n)
    # A modest late-period distribution shift tests temporal generalization.
    late = np.arange(n) >= int(.8 * n)
    distance[late] *= 1.35
    latent = (-6.0 + .7 * np.log1p(amount / 50) + 1.9 * device
              + 1.4 * abroad + .012 * distance + 1.0 * device * abroad)
    labels = rng.binomial(1, 1 / (1 + np.exp(-latent)))
    return pd.DataFrame({
        "transaction_id": [f"tx_{i:08d}" for i in range(n)],
        "timestamp": pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(seconds, unit="s"),
        "customer_id": rng.integers(0, max(30, n // 200), n).astype(str),
        "amount": amount, "distance_km": distance,
        "new_device": device, "international": abroad, "is_fraud": labels,
    })


def validate_events(frame: pd.DataFrame) -> pd.DataFrame:
    required = RAW_COLUMNS + ["is_fraud"]
    if missing := set(required) - set(frame.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = frame[required].copy()
    if df.isna().any().any():
        raise ValueError("Null values are not allowed.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    if df.timestamp.isna().any() or df.transaction_id.duplicated().any():
        raise ValueError("Timestamps must be valid and transaction IDs unique.")
    for col in ["amount", "distance_km"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col]).all() or (df[col] < 0).any():
            raise ValueError(f"{col} must be finite and nonnegative.")
    for col in ["new_device", "international", "is_fraud"]:
        if not df[col].isin([0, 1]).all():
            raise ValueError(f"{col} must contain only 0 or 1.")
        df[col] = df[col].astype(int)
    df["customer_id"] = df.customer_id.astype(str)
    return df.sort_values(["timestamp", "transaction_id"], kind="stable").reset_index(drop=True)


def add_history(frame: pd.DataFrame) -> pd.DataFrame:
    """History contains observed events, never labels. Equal-time events are atomic.

    Test history may include earlier test EVENTS, matching sequential online scoring;
    model fitting and threshold selection never see test labels.
    """
    df = validate_events(frame)
    queues = defaultdict(deque)
    totals = defaultdict(float)
    counts = defaultdict(int)
    recent, means = np.zeros(len(df)), np.zeros(len(df))
    for timestamp, group in df.groupby("timestamp", sort=False):
        for i, row in group.iterrows():
            user = row.customer_id
            q = queues[user]
            while q and (timestamp - q[0]).total_seconds() > 3600:
                q.popleft()
            recent[i] = len(q)
            means[i] = totals[user] / counts[user] if counts[user] else 0.0
        for row in group.itertuples():
            queues[row.customer_id].append(timestamp)
            totals[row.customer_id] += row.amount
            counts[row.customer_id] += 1
    df[HISTORY_COLUMNS[0]], df[HISTORY_COLUMNS[1]] = recent, means
    return df


def split_time(df: pd.DataFrame):
    times = df.timestamp.sort_values().unique()
    if len(times) < 10:
        raise ValueError("Need at least 10 distinct timestamps.")
    first, second = times[int(.6 * len(times))], times[int(.8 * len(times))]
    parts = (df[df.timestamp < first], df[(df.timestamp >= first) & (df.timestamp < second)],
             df[df.timestamp >= second])
    for name, part in zip(["train", "validation", "test"], parts):
        if part.is_fraud.nunique() != 2:
            raise ValueError(f"{name} needs both classes; supply more data.")
    return parts


def load_csv(path: str | Path):
    return pd.read_csv(path, dtype={"customer_id": str, "transaction_id": str})

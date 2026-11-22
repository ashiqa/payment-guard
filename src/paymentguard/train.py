"""Repeatable experiments with a validation-selected model and frozen test policy."""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from paymentguard.data import add_history, load_csv, split_time
from paymentguard.evaluate import choose_threshold, metrics, reliability
from paymentguard.features import FEATURES, transform


def train(csv_path, output="artifacts", seed=42, max_fpr=.01, data_kind="external"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checksum = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()
    df = add_history(load_csv(csv_path))
    train_df, val_df, test_df = split_time(df)
    x_train, x_val, x_test = [transform(part) for part in (train_df, val_df, test_df)]
    candidates = {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)),
        # Explicitly disable random internal validation: our validation is temporal.
        "gradient_boosting": HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15,
            learning_rate=.06, l2_regularization=5, early_stopping=False, random_state=seed),
    }
    validation = {}
    with threadpool_limits(limits=1):
        for name, model in candidates.items():
            model.fit(x_train, train_df.is_fraud)
            scores = model.predict_proba(x_val)[:, 1]
            threshold = choose_threshold(val_df.is_fraud, scores, max_fpr)
            validation[name] = {"threshold": threshold, **metrics(val_df.is_fraud, scores, threshold)}
        winner = max(validation, key=lambda n: (validation[n]["recall"],
            -validation[n]["false_positive_rate"], validation[n]["average_precision"]))
        threshold = validation[winner]["threshold"]
        scores = candidates[winner].predict_proba(x_test)[:, 1]
    code_hash = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))).hexdigest()
    config = {"seed": seed, "max_fpr": max_fpr, "data_sha256": checksum,
              "code_sha256": code_hash, "data_kind": data_kind}
    version = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    report = {
        "model_version": version, "data_kind": data_kind,
        "warning": "Synthetic demo metrics are not evidence of real-world fraud performance." if data_kind == "synthetic" else "Verify provenance, label availability and feature semantics before interpreting results.",
        "config": config, "selected_model": winner, "threshold": threshold,
        "selection_rule": "validation recall, then lower validation FPR, then average precision",
        "environment": {"python": platform.python_version(), **{p: importlib.metadata.version(p)
            for p in ["scikit-learn", "pandas", "numpy", "joblib", "fastapi"]}},
        "splits": {name: {"rows": len(part), "start": str(part.timestamp.min()), "end": str(part.timestamp.max()),
                         "fraud_count": int(part.is_fraud.sum())}
                   for name, part in zip(["train", "validation", "test"], [train_df, val_df, test_df])},
        "validation": validation,
        "test": metrics(test_df.is_fraud, scores, threshold),
        "test_reliability": reliability(test_df.is_fraud, scores),
        "test_slices": {f"{col}={value}": metrics(test_df.loc[test_df[col] == value, "is_fraud"],
                scores[(test_df[col] == value).to_numpy()], threshold)
            for col in ["new_device", "international"] for value in [0, 1]
            if (test_df[col] == value).any()},
    }
    report["test"]["fpr_budget_met"] = report["test"]["false_positive_rate"] <= max_fpr
    artifact = {"model": candidates[winner], "threshold": threshold, "version": version,
                "features": FEATURES, "data_kind": data_kind}
    joblib.dump(artifact, output / "model.joblib")
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report

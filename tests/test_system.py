import copy
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from paymentguard.api import Transaction, create_app
from paymentguard.data import add_history, generate, split_time, validate_events
from paymentguard.evaluate import choose_threshold, metrics
from paymentguard.features import INPUTS, transform
from paymentguard.train import train


class DataTests(unittest.TestCase):
    def test_future_changes_cannot_change_past_features(self):
        raw = generate(1000)
        before = add_history(raw)
        changed = raw.copy()
        changed.loc[500:, "amount"] *= 10
        changed["is_fraud"] = 1 - changed.is_fraud
        after = add_history(changed)
        pd.testing.assert_frame_equal(transform(before.iloc[:500]), transform(after.iloc[:500]))

    def test_history_excludes_current_and_equal_timestamp_events(self):
        raw = generate(500).iloc[:4].copy()
        raw["customer_id"] = "same"
        raw["amount"] = [10, 30, 50, 70]
        raw["timestamp"] = pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z",
                                           "2025-01-01T00:01:00Z", "2025-01-01T02:00:00Z"])
        frame = add_history(raw)
        self.assertEqual(frame.prior_txn_count_1h.tolist(), [0, 0, 2, 0])
        self.assertEqual(frame.prior_mean_amount.tolist(), [0, 0, 20, 30])

    def test_strict_chronology_and_bad_data(self):
        raw = generate(2000)
        a, b, c = split_time(add_history(raw))
        self.assertLess(a.timestamp.max(), b.timestamp.min())
        self.assertLess(b.timestamp.max(), c.timestamp.min())
        for col, value in [("amount", -1), ("distance_km", np.inf), ("is_fraud", 2)]:
            broken = raw.copy()
            broken.loc[0, col] = value
            with self.assertRaises(ValueError):
                validate_events(broken)

    def test_threshold_handles_ties_and_review_none(self):
        y, scores = [0, 0, 1, 1], [.5, .5, .5, .8]
        threshold = choose_threshold(y, scores, 0)
        result = metrics(y, scores, threshold)
        self.assertEqual(result["recall"], .5)
        self.assertEqual(result["false_positive_rate"], 0)
        threshold = choose_threshold(y, [.2] * 4, .01)
        self.assertEqual(metrics(y, [.2] * 4, threshold)["review_rate"], 0)


class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        raw = generate(6000)
        raw.to_csv(cls.root / "data.csv", index=False)
        cls.report = train(cls.root / "data.csv", cls.root / "a", data_kind="synthetic")
        cls.frame = add_history(raw)
        row = cls.frame.iloc[-1]
        cls.payload = {k: row[k].item() if isinstance(row[k], np.generic) else row[k] for k in INPUTS}
        cls.payload.update(transaction_id=row.transaction_id, timestamp=row.timestamp.isoformat())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_reproducible_training_and_test_policy(self):
        repeated = train(self.root / "data.csv", self.root / "b", data_kind="synthetic")
        self.assertEqual(self.report, repeated)
        winner = self.report["selected_model"]
        self.assertEqual(self.report["threshold"], self.report["validation"][winner]["threshold"])
        self.assertLessEqual(self.report["validation"][winner]["false_positive_rate"], .01)

    def test_api_matches_offline_and_has_validation(self):
        model_path = self.root / "a/model.joblib"
        artifact = joblib.load(model_path)
        request_frame = pd.DataFrame([Transaction(**self.payload).model_dump()])
        pd.testing.assert_frame_equal(transform(request_frame).reset_index(drop=True),
                                      transform(self.frame.iloc[[-1]]).reset_index(drop=True))
        with threadpool_limits(limits=1):
            expected = artifact["model"].predict_proba(transform(request_frame))[0, 1]
        with TestClient(create_app(model_path)) as client:
            self.assertEqual(client.get("/ready").status_code, 200)
            response = client.post("/predict", json=self.payload)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertAlmostEqual(response.json()["fraud_score"], expected, places=12)
            self.assertEqual(response.json()["data_kind"], "synthetic")
            for change in [{"amount": -1}, {"international": 3}, {"timestamp": "2025-01-01T00:00:00"},
                           {"is_fraud": 1}, {"prior_txn_count_1h": -1}]:
                payload = copy.deepcopy(self.payload)
                payload.update(change)
                self.assertEqual(client.post("/predict", json=payload).status_code, 422)

    def test_missing_model_is_not_ready(self):
        with TestClient(create_app(self.root / "missing.joblib")) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/ready").status_code, 503)
            self.assertEqual(client.post("/predict", json=self.payload).status_code, 503)


if __name__ == "__main__":
    unittest.main()

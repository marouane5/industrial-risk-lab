import copy
import unittest

import numpy as np

from industrial_risk.pipeline import chronological_split, drift_report, fit_drift_reference, fit_logistic, generate, make_features, metrics, predict, select_threshold


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.rows = generate(seed=7, assets=2, hours=300)
        self.x, self.y, self.meta = make_features(self.rows)

    def test_reproducibility(self):
        self.assertEqual(self.rows, generate(seed=7, assets=2, hours=300))

    def test_labels_use_future_only(self):
        rows = [{"asset": "a", "hour": t, "temperature": 1, "vibration": 1, "pressure": 1, "load": 1, "failure_event": int(t == 20)} for t in range(40)]
        _, y, meta = make_features(rows)
        labels = {m["hour"]: int(v) for m, v in zip(meta, y)}
        self.assertEqual(labels[13], 0)
        self.assertEqual(labels[14], 1)
        self.assertEqual(labels[19], 1)
        self.assertEqual(labels[20], 0)

    def test_future_sensor_changes_do_not_change_past_features(self):
        changed = copy.deepcopy(self.rows)
        for r in changed:
            if r["hour"] > 100:
                r["temperature"] += 10000
        changed_x, _, _ = make_features(changed)
        past = [i for i, m in enumerate(self.meta) if m["hour"] <= 100]
        np.testing.assert_array_equal(self.x[past], changed_x[past])

    def test_labels_never_enter_features(self):
        changed = copy.deepcopy(self.rows)
        for r in changed:
            r["failure_event"] = 1 - r["failure_event"]
        changed_x, changed_y, _ = make_features(changed)
        np.testing.assert_array_equal(self.x, changed_x)
        self.assertFalse(np.array_equal(self.y, changed_y))

    def test_chronological_purge(self):
        train, val, test = chronological_split(self.meta, 300)
        self.assertLess(max(self.meta[i]["label_end_hour"] for i in train), min(self.meta[i]["hour"] for i in val))
        self.assertLess(max(self.meta[i]["label_end_hour"] for i in val), min(self.meta[i]["hour"] for i in test))
        self.assertFalse(set(train).intersection(val))

    def test_training_statistics_are_train_only(self):
        train, val, _ = chronological_split(self.meta, 300)
        model = fit_logistic(self.x[train], self.y[train], iterations=20)
        np.testing.assert_allclose(model["mean"], self.x[train].mean(axis=0))
        scores = predict(model, self.x[val])
        self.assertTrue(((scores > 0) & (scores < 1)).all())

    def test_tied_metrics_prevalence_baseline(self):
        m = metrics([0, 1, 0, 1], [0.2] * 4, 0.3)
        self.assertEqual(m["roc_auc"], 0.5)
        self.assertEqual(m["average_precision"], 0.5)
        self.assertEqual(m["confusion"], {"tp": 0, "fp": 0, "fn": 2, "tn": 2})

    def test_perfect_ranking(self):
        m = metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], 0.5)
        self.assertEqual(m["roc_auc"], 1)
        self.assertEqual(m["average_precision"], 1)
        self.assertEqual(m["f1"], 1)

    def test_drift_reference_stays_fixed(self):
        ref = fit_drift_reference(self.x)
        snapshot = copy.deepcopy(ref)
        same = drift_report(ref, self.x)
        shifted = drift_report(ref, self.x + 100)
        self.assertEqual(snapshot, ref)
        self.assertLess(max(r["psi"] for r in same), 1e-12)
        self.assertGreater(shifted[0]["psi"], 1)

    def test_duplicate_and_gapped_rows_rejected(self):
        with self.assertRaises(ValueError):
            make_features(self.rows + [self.rows[0]])
        with self.assertRaises(ValueError):
            make_features(self.rows[1:100] + self.rows[101:])

    def test_nonfinite_rejected(self):
        broken = copy.deepcopy(self.rows)
        broken[0]["temperature"] = float("nan")
        with self.assertRaises(ValueError):
            make_features(broken)

    def test_batch_scoring_does_not_require_labels(self):
        unlabeled = [{k: v for k, v in r.items() if k != "failure_event"} for r in self.rows]
        x, y, meta = make_features(unlabeled, labeled=False)
        self.assertEqual(x.shape, (578, 16))
        self.assertTrue((y == -1).all())

    def test_threshold_selection_is_deterministic(self):
        self.assertEqual(select_threshold([0, 1], [0.1, 0.9]), select_threshold([0, 1], [0.1, 0.9]))

    def test_invalid_labels_rejected(self):
        for labels in ([1, 2, 1, 2], [0, float("nan"), 0, 1], [[0], [1], [0], [1]]):
            with self.assertRaises(ValueError):
                fit_logistic(np.arange(64).reshape(4, 16), labels, iterations=1)
        with self.assertRaises(ValueError):
            metrics([0, 2], [0.1, 0.9], 0.5)

    def test_drift_wrong_feature_shape_rejected(self):
        ref = fit_drift_reference(self.x)
        with self.assertRaises(ValueError):
            drift_report(ref, np.ones((2, 2)))
        with self.assertRaises(ValueError):
            fit_drift_reference(np.ones((2, 2)))

    def test_inference_rejects_broken_scaler(self):
        model = fit_logistic(self.x, self.y, iterations=1)
        model["std"][0] = 0
        with self.assertRaises(ValueError):
            predict(model, self.x)


if __name__ == "__main__":
    unittest.main()

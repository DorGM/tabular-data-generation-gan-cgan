"""Known-fixture tests for original-domain metrics. No Adult file and no training."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator import (  # noqa: E402
    final_detection,
    fidelity_report,
    hard_validity,
    metrics_schema_document,
    privacy_report,
    research_success,
    schema_from_frame,
    selection_detection,
    utility_report,
)


def reference_frame(n=200):
    age = np.arange(n) % 50 + 20
    return pd.DataFrame(
        {
            "age": age.astype(float),
            "education-num": (np.arange(n) % 10 + 1).astype(float),
            "hours-per-week": (np.arange(n) % 20 + 20).astype(float),
            "capital-gain": np.where(np.arange(n) % 5 == 0, 0.0, 100.0 + (np.arange(n) % 7)),
            "capital-loss": np.where(np.arange(n) % 8 == 0, 0.0, 50.0),
            "fnlwgt": (100000 + np.arange(n) * 3).astype(float),
            "workclass": np.where(np.arange(n) % 17 == 0, "Rare", np.where(np.arange(n) % 2 == 0, "Private", "?")),
            "sex": np.where(np.arange(n) % 2 == 0, "Female", "Male"),
            "income": np.where(np.arange(n) % 4 == 0, 1, 0),
        }
    )


NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "sex"]


class EvaluatorContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = reference_frame()
        cls.schema = schema_from_frame(cls.reference, NUMERIC, CATEGORICAL)

    def test_identical_rows_are_valid_and_close(self):
        report = hard_validity(self.reference, self.schema)
        self.assertTrue(report["valid"])
        self.assertEqual(report["violation_total"], 0)
        distances = fidelity_report(self.reference, self.reference.copy(), self.schema, nn_subsample=50)
        self.assertAlmostEqual(distances["numeric_ks_mean"], 0.0, places=6)
        self.assertAlmostEqual(distances["categorical_tv_mean"], 0.0, places=6)
        self.assertAlmostEqual(distances["correlation_mae"], 0.0, places=6)
        self.assertEqual(distances["category_coverage_mean"], 1.0)

    def test_shifted_and_invalid_rows_are_detected(self):
        shifted = self.reference.copy()
        shifted["age"] = shifted["age"] + 60
        distances = fidelity_report(self.reference, shifted, self.schema, nn_subsample=40)
        self.assertEqual(distances["numeric"]["age"]["ks"], 1.0)

        invalid = self.reference.copy()
        invalid.loc[:9, "capital-gain"] = -50.0
        invalid.loc[10:19, "age"] = 25.4
        invalid.loc[20:29, "workclass"] = "NOT_A_LEVEL"
        invalid.loc[30:34, "education-num"] = np.nan
        report = hard_validity(invalid, self.schema)
        self.assertFalse(report["valid"])
        self.assertGreater(report["violation_counts"]["negative_capital_rows"], 0)
        self.assertGreater(report["violation_counts"]["noninteger_discrete_rows"], 0)
        self.assertGreater(report["violation_counts"]["invalid_category_rows"], 0)
        self.assertGreater(report["violation_counts"]["nonfinite_rows"], 0)

        outside = self.reference.copy()
        outside["age"] = 500.0
        self.assertGreater(hard_validity(outside, self.schema)["violation_counts"]["outside_reference_domain_rows"], 0)

    def test_detection_orientation_and_protocol_b_fit_population(self):
        generator = np.random.default_rng(0)
        same_schema = schema_from_frame(
            pd.DataFrame({"age": np.array([20.0, 60.0]), "sex": ["Female", "Male"]}),
            ["age"],
            ["sex"],
        )
        left = pd.DataFrame({
            "age": generator.normal(40.0, 5.0, size=300),
            "sex": np.where(generator.random(300) < 0.5, "Female", "Male"),
        })
        right = pd.DataFrame({
            "age": generator.normal(40.0, 5.0, size=300),
            "sex": np.where(generator.random(300) < 0.5, "Female", "Male"),
        })
        identical = selection_detection(left, right, same_schema, n_splits=2, n_estimators=30, max_depth=3)
        self.assertLess(abs(identical["mean_auc"] - 0.5), 0.15)
        self.assertFalse(identical["uses_final_test"])

        separated = right.copy()
        separated["age"] = 80.0
        contrast = selection_detection(left, separated, same_schema, n_splits=2, n_estimators=30, max_depth=3)
        self.assertGreater(contrast["mean_auc"], 0.95)
        self.assertAlmostEqual(contrast["oriented_auc"], max(contrast["mean_auc"], 1 - contrast["mean_auc"]))

        seen = {}

        class RecordingForest:
            def __init__(self, *args, **kwargs):
                pass

            def fit(self, features, labels):
                seen["n"] = len(features)
                seen["marker_max"] = float(features[:, 0].max())
                return self

            def predict_proba(self, features):
                score = features[:, 0]
                return np.column_stack([1 - score, score])

        development = self.reference.iloc[:80].copy()
        development_synthetic = development.copy()
        holdout = self.reference.iloc[80:120].copy()
        holdout_synthetic = holdout.copy()
        development["age"] = 0.0
        development_synthetic["age"] = 0.0
        holdout["age"] = 0.0
        holdout_synthetic["age"] = 1.0
        with patch("evaluator.RandomForestClassifier", RecordingForest):
            result = final_detection(
                development, development_synthetic, holdout, holdout_synthetic, self.schema, n_estimators=2,
            )
        self.assertEqual(seen["n"], 160)
        self.assertEqual(seen["marker_max"], 0.0)
        self.assertFalse(result["uses_final_test_for_fitting"])
        self.assertGreater(result["mean_auc"], 0.9)

    def test_utility_refuses_unlabeled_generator_and_unequal_sizes(self):
        matrix = np.column_stack([
            self.reference["age"].to_numpy(),
            self.reference["income"].to_numpy(),
        ])
        labels = self.reference["income"].to_numpy()
        refused = utility_report(matrix[:40], labels[:40], matrix[:40], None, matrix[40:60], labels[40:60], n_estimators=5)
        self.assertFalse(refused["available"])
        with self.assertRaises(ValueError):
            utility_report(matrix[:40], labels[:40], matrix[:30], labels[:30], matrix[40:60], labels[40:60], n_estimators=5)
        scored = utility_report(
            matrix[:80], labels[:80], matrix[:80], labels[:80], matrix[80:120], labels[80:120], n_estimators=10,
        )
        self.assertTrue(scored["available"])
        self.assertAlmostEqual(scored["auc_real"], scored["auc_synthetic"])
        self.assertEqual(scored["prior_baseline_auc"], 0.5)

    def test_privacy_proxy_counts_exact_copies(self):
        synthetic = self.reference.iloc[:10].copy()
        report = privacy_report(synthetic, self.reference, self.schema)
        self.assertEqual(report["exact_duplicate_fraction_vs_reference"], 1.0)
        self.assertTrue(report["does_not_guarantee_privacy"])
        fresh = synthetic.copy()
        fresh["age"] = fresh["age"] + 3
        self.assertEqual(privacy_report(fresh, self.reference, self.schema)["exact_duplicate_count"], 0)

    def test_schema_targets_are_not_a_soft_pass(self):
        document = metrics_schema_document()
        self.assertEqual(document["research_targets"]["protocol_b_detection_auc"], 0.75)
        self.assertEqual(document["research_targets"]["utility_ratio"], 0.95)
        self.assertFalse(research_success(True, 0.99, 0.99)["all_three"])
        self.assertFalse(research_success(True, 0.6, None)["all_three"])
        self.assertTrue(research_success(True, 0.6, 0.96)["all_three"])
        on_disk = ROOT / "results" / "metrics_schema.json"
        self.assertTrue(on_disk.exists())


if __name__ == "__main__":
    unittest.main()

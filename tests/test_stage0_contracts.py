"""Fast CPU contracts for the original Adult GAN code.

Fixtures are synthetic tables. These tests do not read adult.arff and do not train
the published 120-epoch runs.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import (  # noqa: E402
    NUMERIC_COLUMNS,
    ConditionalGenerator,
    Generator,
    _categorical_groups,
    _consider_checkpoint,
    _empty_selection_state,
    decode_rows,
    heldout_detection,
    inverse_numeric,
    load_adult,
    prepare_experiment,
    sample_cgan,
    sample_gan,
    statistical_fidelity,
    train_gan,
)
from repro import config_fingerprint, decoded_selection_age_mean  # noqa: E402


def adult_like_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    income = np.array(["<=50K"] * 300 + [">50K"] * 100)
    rng.shuffle(income)
    work = np.array(["Private", "Self-emp-not-inc", "Local-gov", "?"])
    married = np.array(["Never-married", "Married-civ-spouse", "Divorced"])
    occupation = np.array(["Prof-specialty", "Craft-repair", "?", "Sales"])
    relationship = np.array(["Husband", "Not-in-family", "Own-child"])
    race = np.array(["White", "Black", "Asian-Pac-Islander"])
    sex = np.array(["Male", "Female"])
    country = np.array(["United-States", "Mexico", "?"])
    return pd.DataFrame(
        {
            "age": np.arange(n, dtype=float) % 70 + 18.0,
            "workclass": work[np.arange(n) % len(work)],
            "fnlwgt": 100000.0 + np.arange(n, dtype=float) * 10.0,
            "education": np.array(["HS-grad"] * n),
            "education-num": (np.arange(n) % 10 + 1).astype(float),
            "marital-status": married[np.arange(n) % len(married)],
            "occupation": occupation[np.arange(n) % len(occupation)],
            "relationship": relationship[np.arange(n) % len(relationship)],
            "race": race[np.arange(n) % len(race)],
            "sex": sex[np.arange(n) % len(sex)],
            "capital-gain": 500.0 + np.arange(n, dtype=float),
            "capital-loss": 50.0 + (np.arange(n) % 5).astype(float),
            "hours-per-week": (np.arange(n) % 40 + 20).astype(float),
            "native-country": country[np.arange(n) % len(country)],
            "income": income,
        }
    )


def _fidelity_block(valid=True):
    return {
        "valid": valid,
        "diversity": 1.0,
        "numeric_mean_shift": 0.0,
        "correlation_mae": 0.0,
        "categorical_mae": 0.0,
        "numeric_range_violation_fraction": 0.0,
        "plausible_violation_fraction": 0.0,
        "categorical_extreme_fraction": 0.0,
        "categorical_out_of_unit_fraction": 0.0,
    }


class Stage0Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = adult_like_frame()
        cls.bundle = prepare_experiment(cls.frame, seed=42, output_activation="linear")
        cls.sigmoid = prepare_experiment(cls.frame, seed=42, output_activation="sigmoid")

    def test_import_does_not_load_or_train(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if line.startswith((" ", "\t", "#")):
                continue
            self.assertFalse(stripped.startswith("load_adult("))
            self.assertFalse(stripped.startswith("train_gan("))
            self.assertFalse(stripped.startswith("train_cgan("))
        with self.assertRaises(FileNotFoundError):
            load_adult(ROOT / "data" / "missing-adult.arff")

    def test_splits_are_disjoint_complete_and_shared_by_activation(self):
        bundle = self.bundle
        parts = [bundle["train_index"], bundle["select_index"], bundle["test_index"]]
        combined = np.concatenate(parts)
        self.assertEqual(len(combined), len(self.frame))
        self.assertEqual(len(set(combined.tolist())), len(self.frame))
        for left, right in ((0, 1), (0, 2), (1, 2)):
            self.assertFalse(set(parts[left].tolist()) & set(parts[right].tolist()))
        np.testing.assert_array_equal(bundle["train_index"], self.sigmoid["train_index"])
        np.testing.assert_array_equal(bundle["select_index"], self.sigmoid["select_index"])
        np.testing.assert_array_equal(bundle["test_index"], self.sigmoid["test_index"])
        self.assertEqual(bundle["scaling"], "standard")
        self.assertEqual(self.sigmoid["scaling"], "minmax")

    def test_stratified_income_rates_stay_close(self):
        rates = []
        for key in ("y_train", "y_select", "y_test"):
            labels = self.bundle[key].numpy()
            self.assertEqual(set(np.unique(labels).tolist()), {0.0, 1.0})
            rates.append(float(labels.mean()))
        self.assertLess(max(rates) - min(rates), 0.02)

    def test_scaler_and_categories_are_fit_on_the_fit_slice_only(self):
        bundle = self.bundle
        fit_ages = self.frame.iloc[bundle["train_index"]]["age"].to_numpy(dtype=float)
        np.testing.assert_allclose(bundle["scaler"].mean_[0], fit_ages.mean())
        contaminated = self.frame.copy()
        fit_ids = set(bundle["train_index"].tolist())
        for index in range(len(contaminated)):
            if index not in fit_ids:
                contaminated.loc[index, "age"] = 5000.0
        refit = prepare_experiment(contaminated, seed=42, output_activation="linear")
        np.testing.assert_allclose(bundle["scaler"].mean_, refit["scaler"].mean_)

        test_only = self.frame.copy()
        test_only.loc[int(bundle["test_index"][0]), "workclass"] = "ONLY_IN_TEST"
        hidden = prepare_experiment(test_only, seed=42, output_activation="linear")
        self.assertEqual(bundle["feature_names"], hidden["feature_names"])
        self.assertNotIn("workclass_ONLY_IN_TEST", hidden["feature_names"])

        fit_only = self.frame.copy()
        fit_only.loc[int(bundle["train_index"][0]), "workclass"] = "ONLY_IN_FIT"
        revealed = prepare_experiment(fit_only, seed=42, output_activation="linear")
        self.assertIn("workclass_ONLY_IN_FIT", revealed["feature_names"])

    def test_inverse_numeric_restores_fit_rows(self):
        for bundle in (self.bundle, self.sigmoid):
            restored = inverse_numeric(bundle["x_train"], bundle["feature_names"], bundle["scaler"])
            original = self.frame.iloc[bundle["train_index"]][list(NUMERIC_COLUMNS)].to_numpy(dtype=float)
            np.testing.assert_allclose(restored, original, atol=1e-4, rtol=1e-5)

    def test_categorical_groups_and_argmax(self):
        groups = _categorical_groups(
            ["ab_z", "a_q", "age"],
            ["a", "ab"],
        )
        self.assertEqual([group["name"] for group in groups], ["ab", "a"])
        self.assertEqual(groups[0]["labels"], ["z"])
        self.assertEqual(groups[1]["labels"], ["q"])

        bundle = self.bundle
        decoded = decode_rows(
            bundle["x_train"][:20],
            bundle["feature_names"],
            bundle["numeric_columns"],
            bundle["categorical_groups"],
            bundle["scaler"],
        )
        start = 0
        raw = bundle["x_train"][:20].numpy()
        for group in bundle["categorical_groups"]:
            width = len(group["indexes"])
            self.assertTrue(np.allclose(decoded["hard"][:, start:start + width].sum(axis=1), 1.0))
            self.assertTrue(np.allclose(decoded["hard"][:, start:start + width], raw[:, group["indexes"]]))
            start += width

        array = raw[:2].copy()
        group = bundle["categorical_groups"][0]
        array[0, group["indexes"]] = 0.0
        array[1, group["indexes"]] = 0.0
        array[1, group["indexes"][0]] = 0.2
        array[1, group["indexes"][-1]] = 3.0
        special = decode_rows(
            array, bundle["feature_names"], bundle["numeric_columns"],
            bundle["categorical_groups"], bundle["scaler"],
        )
        width = len(group["indexes"])
        self.assertTrue(np.allclose(special["hard"][0, :width], 0.0))
        self.assertEqual(float(special["hard"][1, width - 1]), 1.0)
        self.assertGreater(special["categorical_extreme_fraction"], 0.0)

    def test_invalid_range_and_collapse_are_rejected(self):
        bundle = self.bundle
        decoded = decode_rows(
            bundle["x_select"], bundle["feature_names"], bundle["numeric_columns"],
            bundle["categorical_groups"], bundle["scaler"],
        )
        same = statistical_fidelity(decoded, decoded, bundle["numeric_columns"])
        self.assertTrue(same["valid"])
        self.assertEqual(same["plausible_violation_fraction"], 0.0)

        shifted = {
            "numeric": decoded["numeric"].copy(),
            "hard": decoded["hard"].copy(),
            "finite": True,
            "categorical_extreme_fraction": 0.0,
            "categorical_out_of_unit_fraction": 0.0,
        }
        shifted["numeric"][:, NUMERIC_COLUMNS.index("capital-gain")] = -5.0
        poor = statistical_fidelity(decoded, shifted, bundle["numeric_columns"])
        self.assertFalse(poor["valid"])
        self.assertGreater(poor["plausible_violation_fraction"], 0.02)

        collapsed = {
            "numeric": np.repeat(decoded["numeric"][:1], len(decoded["numeric"]), axis=0),
            "hard": np.repeat(decoded["hard"][:1], len(decoded["hard"]), axis=0),
            "finite": True,
            "categorical_extreme_fraction": 0.0,
            "categorical_out_of_unit_fraction": 0.0,
        }
        rejected = statistical_fidelity(decoded, collapsed, bundle["numeric_columns"])
        self.assertFalse(rejected["valid"])
        self.assertTrue(rejected["collapsed"])

    def test_selection_age_mean_uses_the_full_slice(self):
        encoded = self.bundle["x_select"].clone()
        encoded[:8, 0] = 6.0
        encoded[8:, 0] = -2.0
        viewed = dict(self.bundle)
        viewed["x_select"] = encoded
        full = decoded_selection_age_mean(viewed)
        prefix = decode_rows(
            encoded[:8], viewed["feature_names"], viewed["numeric_columns"],
            viewed["categorical_groups"], viewed["scaler"],
        )
        self.assertEqual(encoded.shape[0], decode_rows(
            encoded, viewed["feature_names"], viewed["numeric_columns"],
            viewed["categorical_groups"], viewed["scaler"],
        )["numeric"].shape[0])
        self.assertGreater(abs(full - float(prefix["numeric"][:, 0].mean())), 1.0)

    def test_shapes_labels_and_deterministic_sampling(self):
        width = 6
        generator = Generator(4, width, hidden_dim=8, output_activation="sigmoid")
        first = sample_gan(generator, 5, 4, seed=3)
        second = sample_gan(generator, 5, 4, seed=3)
        third = sample_gan(generator, 5, 4, seed=4)
        self.assertEqual(tuple(first.shape), (5, width))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, third))
        self.assertTrue(torch.all((first >= 0) & (first <= 1)))

        conditional = ConditionalGenerator(4, 2, width, hidden_dim=8, output_activation="linear")
        labels = torch.tensor([0, 1, 1, 0])
        features, returned = sample_cgan(conditional, labels, 4, 2, seed=5)
        other, _ = sample_cgan(conditional, torch.ones(4, dtype=torch.long), 4, 2, seed=5)
        self.assertEqual(tuple(features.shape), (4, width))
        self.assertTrue(torch.equal(returned, labels))
        self.assertFalse(torch.equal(features, other))
        with self.assertRaises(ValueError):
            sample_cgan(conditional, torch.tensor([0, 2]), 4, 2, seed=1)

    def test_train_and_eval_modes(self):
        generator = Generator(4, 3, hidden_dim=32, dropout=0.8, output_activation="linear")
        generator.eval()
        baseline = generator(torch.ones(20, 4))
        generator.train()
        differed = any(
            not torch.allclose(generator(torch.ones(20, 4)), baseline)
            for _ in range(8)
        )
        self.assertTrue(differed)
        modes = []
        config = {
            "latent_dim": 4,
            "batch_size": 8,
            "hidden_dim": 32,
            "dropout": 0.0,
            "output_activation": "linear",
            "optimizer": "Adam",
            "lr": 1e-3,
            "weight_decay": 0.0,
            "epochs": 1,
        }

        def factory(*args, **kwargs):
            module = Generator(*args, **kwargs)
            module.register_forward_pre_hook(lambda trained, _inputs: modes.append(bool(trained.training)))
            return module

        with patch("main.Generator", factory):
            result = train_gan(torch.randn(24, 3), config, seed=0, device="cpu", selection=None, log_every=0)
        self.assertIn(True, modes)
        self.assertFalse(result["generator"].training)
        sample_gan(result["generator"], 2, 4, seed=0)
        self.assertFalse(result["generator"].training)
        self.assertIsInstance(result["generator"].net[1], nn.LeakyReLU)
        conditional = ConditionalGenerator(4, 2, 3, hidden_dim=8)
        self.assertIsInstance(conditional.net[1], nn.ReLU)

    def test_checkpoint_rule_minimizes_gap_among_eligible(self):
        generator = Generator(4, 3, hidden_dim=8)
        state = _empty_selection_state()
        selection = {"every": 1, "x_select": torch.zeros(4, 3), "bundle": {}, "n_splits": 2, "n_estimators": 2}
        config = {"epochs": 3, "latent_dim": 4}
        reports = [
            {"mean_auc": 1.0, "fold_auc": [1.0], "gap": 0.50, "fidelity": _fidelity_block(True)},
            {"mean_auc": 0.9, "fold_auc": [0.9], "gap": 0.40, "fidelity": _fidelity_block(True)},
            {"mean_auc": 0.5, "fold_auc": [0.5], "gap": 0.00, "fidelity": _fidelity_block(False)},
        ]
        with patch("main._sample_for_selection", return_value=(torch.zeros(4, 3), None)), patch(
            "main.compare_real_synthetic", side_effect=reports
        ):
            for epoch in range(3):
                _consider_checkpoint(generator, epoch, "gan", None, config, selection, state, "cpu")
        self.assertEqual(state["best_epoch"], 1)
        self.assertAlmostEqual(state["best_gap"], 0.40)

    def test_heldout_detection_fits_on_the_combined_pool(self):
        real = np.zeros((40, 3))
        synth = np.ones((40, 3))
        seen = []

        class RecordingForest:
            def __init__(self, *args, **kwargs):
                pass

            def fit(self, features, labels):
                seen.append((len(features), sorted(set(np.asarray(labels).tolist()))))
                return self

            def predict_proba(self, features):
                return np.column_stack([1 - features[:, 0], features[:, 0]])

        with patch("main.RandomForestClassifier", RecordingForest):
            result = heldout_detection(real, synth, random_state=0, n_estimators=2, repeats=2, test_size=0.5)
        self.assertEqual(len(seen), 2)
        for n_rows, labels in seen:
            self.assertEqual(n_rows, 40)
            self.assertEqual(labels, [0.0, 1.0])
        self.assertGreater(result["mean_auc"], 0.9)

    def test_published_docs_do_not_repeat_the_old_errors(self):
        # The README now describes the current ladder only; the baseline study lives in the notebook.
        notebook = (ROOT / "experiments.ipynb").read_text(encoding="utf-8")
        self.assertIn("full_selection_age_mean", notebook)
        self.assertIn("not the first eligible checkpoint", notebook)
        self.assertNotIn("decoded_age_mean", notebook)

    def test_config_fingerprint_is_order_independent(self):
        left = config_fingerprint({"b": 1, "a": [1, 2]})
        right = config_fingerprint({"a": [1, 2], "b": 1})
        self.assertEqual(left, right)
        self.assertNotEqual(left, config_fingerprint({"a": [1, 2], "b": 2}))


if __name__ == "__main__":
    unittest.main()

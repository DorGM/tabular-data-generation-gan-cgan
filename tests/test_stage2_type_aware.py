"""Shape, gradient, and hard-sample contracts for the type-aware generator."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator import frame_from_legacy_decoded, hard_validity, schema_from_frame  # noqa: E402
from main import decode_rows, prepare_experiment  # noqa: E402
from type_aware import (  # noqa: E402
    TypeAwareConditionalGenerator,
    fit_type_spec,
    gumbel_softmax,
    temperature_at,
    train_type_aware_cgan,
)


def small_frame(n=160):
    return pd.DataFrame(
        {
            "age": (np.arange(n) % 40 + 20).astype(float),
            "workclass": np.where(np.arange(n) % 2 == 0, "Private", "?"),
            "fnlwgt": (10000 + np.arange(n) * 5).astype(float),
            "education": np.array(["HS-grad"] * n),
            "education-num": (np.arange(n) % 5 + 1).astype(float),
            "marital-status": np.where(np.arange(n) % 3 == 0, "Never-married", "Married-civ-spouse"),
            "occupation": np.where(np.arange(n) % 2 == 0, "Sales", "Craft-repair"),
            "relationship": np.where(np.arange(n) % 2 == 0, "Husband", "Not-in-family"),
            "race": np.array(["White"] * n),
            "sex": np.where(np.arange(n) % 2 == 0, "Female", "Male"),
            "capital-gain": np.where(np.arange(n) % 4 == 0, 0.0, 250.0),
            "capital-loss": np.where(np.arange(n) % 5 == 0, 0.0, 40.0),
            "hours-per-week": (np.arange(n) % 20 + 20).astype(float),
            "native-country": np.array(["United-States"] * n),
            "income": np.where(np.arange(n) % 4 == 0, ">50K", "<=50K"),
        }
    )


class TypeAwareContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = small_frame()
        cls.bundle = prepare_experiment(cls.frame, seed=1, output_activation="sigmoid")
        cls.spec = fit_type_spec(cls.frame, cls.bundle)
        cls.model = TypeAwareConditionalGenerator(6, 2, cls.spec, hidden_dim=16, dropout=0.0)

    def _labels(self, n_rows, value):
        labels = torch.zeros(n_rows, 2)
        labels[:, value] = 1.0
        return labels

    def test_temperature_schedule_is_documented_endpoints(self):
        self.assertEqual(temperature_at(0, 120), 1.0)
        self.assertAlmostEqual(temperature_at(119, 120), 0.5)

    def test_soft_train_and_hard_eval_categories(self):
        noise = torch.randn(32, 6)
        labels = self._labels(32, 0)
        self.model.train()
        soft = self.model(noise, labels, temperature=2.0, hard=False)
        for group in self.spec["groups"]:
            block = soft[:, group["indexes"]]
            self.assertTrue(torch.allclose(block.sum(dim=1), torch.ones(32), atol=1e-5))
        relaxed = gumbel_softmax(torch.zeros(64, 4), temperature=5.0, hard=False)
        self.assertGreater(float((relaxed.max(dim=1).values < 0.7).float().mean()), 0.8)
        self.model.eval()
        hard = self.model(noise, labels, temperature=0.5, hard=True)
        again = self.model(noise, labels, temperature=0.5, hard=True)
        self.assertTrue(torch.equal(hard, again))
        for group in self.spec["groups"]:
            block = hard[:, group["indexes"]]
            self.assertTrue(torch.equal(block.sum(dim=1), torch.ones(32)))
            self.assertTrue(torch.all((block == 0) | (block == 1)))

    def test_conditioning_and_gradients(self):
        noise = torch.randn(16, 6)
        low = self.model(noise, self._labels(16, 0), temperature=1.0, hard=False)
        high = self.model(noise, self._labels(16, 1), temperature=1.0, hard=False)
        self.assertFalse(torch.allclose(low, high))
        noise = torch.randn(16, 6, requires_grad=True)
        output = self.model(noise, self._labels(16, 1), temperature=1.0, hard=False)
        output.sum().backward()
        self.assertIsNotNone(noise.grad)
        self.assertTrue(torch.isfinite(noise.grad).all())
        self.assertGreater(float(noise.grad.abs().sum()), 0.0)

    def test_straight_through_is_one_hot_and_differentiable(self):
        logits = torch.tensor([[0.2, 0.1, 1.5]], requires_grad=True)
        sample = gumbel_softmax(logits, temperature=1.0, hard=True, straight_through=True)
        self.assertTrue(torch.all((sample == 0) | (sample == 1)))
        self.assertEqual(float(sample.sum()), 1.0)
        sample.sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_hard_samples_pass_structural_validity(self):
        self.model.eval()
        labels = torch.zeros(64, 2)
        labels[:, 0] = 1.0
        encoded = self.model(torch.randn(64, 6), labels, hard=True)
        decoded = decode_rows(
            encoded, self.bundle["feature_names"], self.bundle["numeric_columns"],
            self.bundle["categorical_groups"], self.bundle["scaler"],
        )
        table = frame_from_legacy_decoded(decoded, self.bundle)
        schema = schema_from_frame(
            self.frame,
            self.bundle["numeric_columns"],
            self.bundle["categorical_columns"],
        )
        report = hard_validity(table, schema)
        self.assertTrue(report["valid"], report)
        age = table["age"].to_numpy()
        hours = table["hours-per-week"].to_numpy()
        education = table["education-num"].to_numpy()
        self.assertTrue(np.allclose(age, np.round(age)))
        self.assertTrue(np.allclose(hours, np.round(hours)))
        self.assertTrue(np.allclose(education, np.round(education)))
        self.assertTrue(np.all(table["capital-gain"].to_numpy() >= -1e-2))
        self.assertTrue(np.all(table["capital-loss"].to_numpy() >= -1e-2))

    def test_short_training_keeps_finite_gradients_and_eval_mode(self):
        config = {
            "latent_dim": 6,
            "batch_size": 16,
            "hidden_dim": 16,
            "dropout": 0.0,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "optimizer": "Adam",
            "epochs": 1,
        }
        result = train_type_aware_cgan(
            self.bundle["x_train"], self.bundle["y_train"], self.spec, config,
            seed=0, device="cpu", log_every=0,
        )
        self.assertEqual(len(result["d_loss"]), 1)
        self.assertTrue(np.isfinite(result["d_loss"][0]))
        self.assertTrue(np.isfinite(result["g_loss"][0]))
        self.assertFalse(result["generator"].training)
        self.assertEqual(result["kind"], "type_aware_cgan")


if __name__ == "__main__":
    unittest.main()

"""Fast CPU contracts for the diffusion steps (7-9). Synthetic fixtures; no data file, no training."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion import (  # noqa: E402
    STEPS,
    EDMDiffusion,
    GaussianMultinomialDiffusion,
    TabEncoder,
    VAE,
    cosine_alphas_cumprod,
)
from ladder import CATEGORICAL, NUMERIC, expected_violations, rule_violations  # noqa: E402


def fixture_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    rel = rng.choice(["Husband", "Wife", "Not-in-family", "Own-child"], n)
    sex = np.where(rel == "Husband", "Male", np.where(rel == "Wife", "Female", rng.choice(["Male", "Female"], n)))
    marital = np.where(np.isin(rel, ["Husband", "Wife"]), "Married-civ-spouse",
                       rng.choice(["Never-married", "Divorced"], n))
    workclass = rng.choice(["Private", "Self-emp", "?", "Never-worked"], n, p=[0.6, 0.3, 0.07, 0.03])
    occupation = np.where(np.isin(workclass, ["?", "Never-worked"]), "?", rng.choice(["Sales", "Tech-support"], n))
    return pd.DataFrame({
        "age": rng.integers(17, 90, n).astype(float),
        "fnlwgt": rng.integers(20000, 900000, n).astype(float),
        "education-num": rng.integers(1, 17, n).astype(float),
        "capital-gain": np.where(rng.random(n) < 0.9, 0.0, rng.integers(100, 99999, n)).astype(float),
        "capital-loss": np.where(rng.random(n) < 0.95, 0.0, rng.integers(100, 4000, n)).astype(float),
        "hours-per-week": rng.integers(1, 99, n).astype(float),
        "workclass": workclass, "marital-status": marital, "occupation": occupation, "relationship": rel,
        "race": rng.choice(["White", "Black"], n), "sex": sex,
        "native-country": rng.choice(["United-States", "Mexico"], n),
    })


class ScheduleAndPosterior(unittest.TestCase):
    def test_cosine_schedule(self):
        ab = cosine_alphas_cumprod(1000)
        self.assertEqual(len(ab), 1001)
        self.assertAlmostEqual(ab[0], 1.0)
        self.assertTrue(np.all(np.diff(ab) <= 0))
        self.assertLess(ab[-1], 1e-3)

    def test_categorical_posterior_matches_bayes(self):
        """q(x_s | x_t, x0) from the closed form equals Bayes' rule on the two forward steps."""
        torch.manual_seed(0)
        model = GaussianMultinomialDiffusion(2, [3, 5], STEPS["7_ddpm"])
        ab = model.alpha_bar
        k = 5
        for t, s in [(10, 9), (500, 300), (1000, 0)]:
            ab_t, ab_s = ab[t].view(1, 1), ab[s].view(1, 1)
            x0 = F.one_hot(torch.tensor([2]), k).float()
            xt = F.one_hot(torch.tensor([4]), k).float()
            post = model._cat_posterior(xt, x0, ab_t, ab_s, k)
            alpha_ts = ab_t / ab_s
            q_t_given_s = alpha_ts * torch.eye(k) + (1 - alpha_ts) / k   # row: x_s, column: x_t
            q_s_given_0 = ab_s * x0 + (1 - ab_s) / k
            bayes = q_s_given_0[0] * q_t_given_s[:, 4]
            bayes = bayes / bayes.sum()
            self.assertTrue(torch.allclose(post[0], bayes, atol=1e-6), (t, s, post, bayes))
            self.assertAlmostEqual(float(post.sum()), 1.0, places=5)

    def test_loss_and_sampling_shapes(self):
        torch.manual_seed(0)
        model = GaussianMultinomialDiffusion(2, [3, 5], dict(STEPS["7_ddpm"], hidden=(16,), dim_t=8))
        x_num = torch.randn(7, 2)
        x_cat = torch.stack([torch.randint(3, (7,)), torch.randint(5, (7,))], 1)
        y = torch.randint(2, (7,))
        loss = model.per_row_loss(x_num, x_cat, y, torch.randint(1, 1001, (7,)))
        self.assertEqual(loss.shape, (7,))
        self.assertTrue(torch.isfinite(loss).all())
        g = torch.Generator().manual_seed(1)
        num, cat = model.sample(y, 5, g)
        self.assertEqual(num.shape, (7, 2))
        self.assertTrue((cat[:, 0] < 3).all() and (cat[:, 1] < 5).all())


class EncoderAndRules(unittest.TestCase):
    def test_round_trip_on_fit_rows(self):
        frame = fixture_frame()
        enc = TabEncoder(0).fit(frame)
        x_num, x_cat = enc.transform(frame)
        back = enc.inverse(x_num, x_cat)
        for column in CATEGORICAL:
            self.assertTrue((back[column].to_numpy() == frame[column].astype(str).to_numpy()).all(), column)
        for column in NUMERIC:
            close = np.abs(back[column].to_numpy() - frame[column].to_numpy()) <= 1.0 + 1e-6 * frame[column].abs()
            self.assertGreater(close.mean(), 0.97, column)
        # Exact zeros survive through the gate.
        for column in ("capital-gain", "capital-loss"):
            self.assertTrue(((back[column] == 0) == (frame[column] == 0)).all())

    def test_expected_violations_counts_hard_rows(self):
        frame = fixture_frame()
        frame.loc[0, "sex"] = "Female"          # a husband who is not male
        frame.loc[1, "workclass"] = "Private"   # occupation missing without a reason
        frame.loc[1, "occupation"] = "?"
        enc = TabEncoder(0).fit(frame)
        _x_num, x_cat = enc.transform(frame)
        probs = [F.one_hot(torch.tensor(x_cat[:, j]), k).float() for j, k in enumerate(enc.num_classes)]
        expected = expected_violations(enc.cat_probs(probs), enc.levels)
        mask, _counts = rule_violations(frame)
        self.assertTrue(np.array_equal(expected.numpy() > 0, mask))


class LatentModel(unittest.TestCase):
    def test_vae_and_edm_shapes(self):
        torch.manual_seed(0)
        config = dict(STEPS["9_latent_rules"], hidden=(16,), dim_t=8, vae_ff=8)
        vae = VAE(2, [3, 5], config)
        x_num = torch.randn(6, 2)
        x_cat = torch.stack([torch.randint(3, (6,)), torch.randint(5, (6,))], 1)
        loss, recon = vae.loss(x_num, x_cat, 1e-2)
        self.assertTrue(torch.isfinite(loss))
        dim = vae.n_tokens * vae.d
        edm = EDMDiffusion(dim, config)
        y = torch.randint(2, (6,))
        self.assertEqual(edm.per_row_loss(torch.randn(6, dim), y).shape, (6,))
        z = edm.sample(y, dim, 4, torch.Generator().manual_seed(0))
        num, logits = vae.decode(z)
        self.assertEqual(num.shape, (6, 2))
        self.assertEqual([lg.shape[1] for lg in logits], [3, 5])


if __name__ == "__main__":
    unittest.main()

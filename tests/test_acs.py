"""ACS protocol contracts on synthetic fixtures (no data file): household splits and rules."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import acs_data  # noqa: E402


def fake_people(n_households=4000, seed=0):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 6, n_households)
    serial = np.repeat([f"H{i:06d}" for i in range(n_households)], sizes)
    n = len(serial)
    rel = rng.choice(["0", "1", "2", "13", "11"], n)
    mar = np.where(rel == "1", "1", np.where(rel == "13", rng.choice(["3", "5"], n), rng.choice(["1", "5"], n)))
    occ = rng.choice(["10", "4700", "9800", "9830"], n, p=[0.5, 0.45, 0.03, 0.02])
    cow = np.where(np.isin(occ, acs_data.MILITARY), "5", rng.choice(["1", "2", "6"], n))
    return pd.DataFrame({"SERIALNO": serial, "AGEP": rng.integers(17, 90, n).astype(float),
                         "WKHP": rng.integers(1, 99, n).astype(float), "COW": cow, "SCHL": rng.choice(["16", "21"], n),
                         "MAR": mar, "OCCP": occ, "POBP": rng.choice(["6", "303"], n), "RELP": rel,
                         "SEX": rng.choice(["1", "2"], n), "RAC1P": rng.choice(["1", "2"], n),
                         "income": rng.integers(0, 2, n)})


class Splits(unittest.TestCase):
    def setUp(self):
        self.frame = fake_people()
        self.sizes = (("train", 3000), ("validation", 800), ("test", 3000), ("reference", 1000))
        self._old, acs_data.PART_SIZES = acs_data.PART_SIZES, self.sizes

    def tearDown(self):
        acs_data.PART_SIZES = self._old

    def test_households_never_cross_parts_and_rows_cover_everything(self):
        parts = acs_data.make_splits(self.frame)
        households = {k: set(self.frame["SERIALNO"].iloc[v]) for k, v in parts.items()}
        names = list(parts)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                self.assertFalse(households[a] & households[b], (a, b))
        rows = np.concatenate(list(parts.values()))
        self.assertEqual(len(rows), len(self.frame))
        self.assertEqual(len(set(rows.tolist())), len(self.frame))
        for name, target in self.sizes:
            self.assertGreaterEqual(len(parts[name]), target)
            self.assertLess(len(parts[name]), target + 6)  # overshoot is at most one household

    def test_split_is_deterministic(self):
        a, b = acs_data.make_splits(self.frame), acs_data.make_splits(self.frame)
        self.assertEqual(acs_data.split_digest(a), acs_data.split_digest(b))


class Rules(unittest.TestCase):
    def test_clean_fixture_has_no_violations_and_each_rule_fires(self):
        frame = fake_people(500)
        mask, counts = acs_data.rule_violations(frame)
        self.assertFalse(mask.any(), counts)
        broken = frame.head(3).copy()
        broken.loc[broken.index[0], ["RELP", "MAR"]] = ["1", "5"]
        broken.loc[broken.index[1], ["RELP", "MAR"]] = ["13", "1"]
        broken.loc[broken.index[2], ["OCCP", "COW"]] = ["9800", "1"]
        _mask, counts = acs_data.rule_violations(broken)
        self.assertEqual(counts, {"spouse_married": 1, "partner_unmarried": 1, "military_federal": 1})

    def test_expected_violations_matches_hard_rules_on_one_hots(self):
        frame = fake_people(300)
        frame.loc[0, ["RELP", "MAR"]] = ["1", "5"]
        frame.loc[1, ["RELP", "MAR"]] = ["13", "1"]
        frame.loc[2, ["OCCP", "COW"]] = ["9830", "2"]
        levels = {c: sorted(frame[c].unique().tolist()) for c in ("RELP", "MAR", "OCCP", "COW")}
        probs = {c: F.one_hot(torch.tensor([levels[c].index(v) for v in frame[c]]), len(levels[c])).float()
                 for c in levels}
        expected = acs_data.expected_violations(probs, levels)
        mask, _ = acs_data.rule_violations(frame)
        self.assertTrue(np.array_equal(expected.numpy() > 0, mask))


class Resume(unittest.TestCase):
    def test_gan_resume_is_identical_to_an_uninterrupted_run(self):
        import subprocess
        out = subprocess.run([sys.executable, str(ROOT / "tests" / "resume_check.py")], capture_output=True,
                             text=True, timeout=600)
        self.assertIn("IDENTICAL", out.stdout, out.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()

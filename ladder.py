"""Improvement ladder for the Adult conditional GAN.

Each rung adds one change, or a named bundle of related changes, on top of the
previous rung. Bundled changes (rungs 1, 3, 4 and 5) are measured together, not
one by one:

  0 baseline    one sigmoid head over the whole row, BCE, uniform fake labels,
                argmax decode, no rounding (mirrors the published sigmoid cGAN)
  1 sampling    sample categories instead of argmax, zero-inflated capital
                columns, integer rounding, fake labels at the fit base rate
  2 st_onehot   straight-through Gumbel one-hots, so the discriminator sees
                hard 0/1 categories during training as it does for real rows
  3 wgan_gp     Wasserstein loss with gradient penalty, 5 critic steps, PacGAN
  4 ctgan_norm  mode-specific normalization (Bayesian GMM) for continuous
                columns and training-by-sampling of categorical conditions
  5 rules       domain consistency rules (relationship / sex / marital status,
                and the two linked missing-value fields) as a loss penalty,
                plus rejection of rule-breaking rows at sampling time

A sixth variant, rung 5 plus snapping capital values to common real values, is
post-processing only and lives in evaluate_holdout.py.

Model selection uses the selection slice only (protocol A). The test split is
scored once per run, after selection (protocol B detection and TSTR utility).
Importing this module does not read data or train.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import BayesianGaussianMixture

from evaluator import (
    _design_matrix,
    final_detection,
    fidelity_report,
    hard_validity,
    schema_from_frame,
    selection_detection,
    utility_report,
)
from main import INCOME_MAP, load_adult, prepare_experiment

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "ladder"
MODEL_DIR = ROOT / "models" / "ladder"
NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
INTEGER = {"age", "education-num", "hours-per-week", "fnlwgt", "capital-gain", "capital-loss"}
ZERO_INFLATED = {"capital-gain", "capital-loss"}
SEEDS = (42, 68, 124)
# TABGEN_DATASET=acs switches columns, rules and output folders to ACS California 2018
# (acs_data.py, PROTOCOL.md). Adult stays the default, so nothing changes for it.
DATASET = os.environ.get("TABGEN_DATASET", "adult")
if DATASET == "acs":
    import acs_data

    NUMERIC, CATEGORICAL = list(acs_data.NUMERIC), list(acs_data.CATEGORICAL)
    INTEGER, ZERO_INFLATED = set(acs_data.INTEGER), set()
    OUT_DIR, MODEL_DIR = ROOT / "results" / "acs" / "train", ROOT / "models" / "acs"

BASE = {
    "epochs": 120, "latent_dim": 64, "hidden_dim": 128, "batch_size": 64, "lr": 1e-3,
    "betas": (0.5, 0.999), "loss": "bce", "critic_steps": 1, "pac": 1, "gp_weight": 10.0,
    "norm": "minmax", "zero_inflated": False, "round_int": False,
    "cat_train": "sigmoid", "cat_sample": "argmax", "label_prior": "uniform",
    "tbs": False, "temperature": 0.2, "score_every": 10,
}
RUNGS = {
    "0_baseline": dict(BASE),
    "1_sampling": dict(BASE, cat_train="gumbel_soft", cat_sample="sample", zero_inflated=True,
                       round_int=True, label_prior="empirical"),
}
RUNGS["2_st_onehot"] = dict(RUNGS["1_sampling"], cat_train="gumbel_st")
RUNGS["3_wgan_gp"] = dict(RUNGS["2_st_onehot"], loss="wgangp", critic_steps=5, pac=10, batch_size=500,
                          lr=2e-4, betas=(0.5, 0.9), hidden_dim=256, epochs=300, score_every=25)
RUNGS["4_ctgan_norm"] = dict(RUNGS["3_wgan_gp"], norm="gmm", tbs=True)
RUNGS["5_rules"] = dict(RUNGS["4_ctgan_norm"], rule_weight=1.0, reject=True)

# Domain rules. They come from what the fields mean, not from fitted numbers:
# a husband is a married man, a wife is a married woman, and occupation is
# missing exactly when the person has no job to describe (workclass missing or
# never worked). In all 48,842 Adult rows the second and third rules have no
# exceptions and the first has 4 (recording errors).
MARRIED = ("Married-civ-spouse", "Married-AF-spouse")
NO_OCCUPATION = ("?", "Never-worked")


def rule_violations(frame):
    """Boolean mask of rows that break a domain rule, plus counts per rule."""
    rel, sex = frame["relationship"].astype(str), frame["sex"].astype(str)
    marital = frame["marital-status"].astype(str)
    workclass, occupation = frame["workclass"].astype(str), frame["occupation"].astype(str)
    rules = {
        "spouse_sex": ((rel == "Husband") & (sex != "Male")) | ((rel == "Wife") & (sex != "Female")),
        "spouse_married": rel.isin(["Husband", "Wife"]) & ~marital.isin(MARRIED),
        "missing_link": (occupation == "?") != workclass.isin(NO_OCCUPATION),
    }
    mask = np.zeros(len(frame), dtype=bool)
    for value in rules.values():
        mask |= value.to_numpy()
    return mask, {name: int(value.sum()) for name, value in rules.items()}


if DATASET == "acs":
    rule_violations = acs_data.rule_violations  # noqa: F811


# ---------------------------------------------------------------- data ----

def split_indices(frame, seed):
    """Row positions (in load_adult order) of the fit, selection and test slices for a split seed."""
    bundle = prepare_experiment(frame, seed=seed, output_activation="sigmoid")
    return {name: np.asarray(bundle[f"{name}_index"], dtype=int) for name in ("train", "select", "test")}


def split_frames(frame, seed):
    """Same fit / selection / test rows as the published experiment."""
    bundle = prepare_experiment(frame, seed=seed, output_activation="sigmoid")
    data = frame.drop(columns=["education"]).copy()
    data["income"] = data["income"].map(INCOME_MAP).astype(int)
    take = lambda index: data.iloc[np.asarray(index)].reset_index(drop=True)
    return take(bundle["train_index"]), take(bundle["select_index"]), take(bundle["test_index"])


class Transformer:
    """Row encoder fit on the fit slice only. Spans are (kind, width, column)."""

    def __init__(self, config, seed):
        self.config = config
        self.seed = seed

    def fit(self, frame):
        self.spans, self.info = [], {}
        for column in NUMERIC:
            values = frame[column].to_numpy(dtype=float)
            info = {"min": float(values.min()), "max": float(values.max())}
            if self.config["zero_inflated"] and column in ZERO_INFLATED:
                info["zero"] = True
                self.spans.append(("cat", 2, column + "__zero"))
                values = values[values > 0]
                values = np.log1p(values)
                info["log"] = True
                info["lmin"], info["lmax"] = float(values.min()), float(values.max())
            if self.config["norm"] == "gmm":
                source = np.log1p(values) if column == "fnlwgt" else values
                if column == "fnlwgt":
                    info["log"] = True
                gmm = BayesianGaussianMixture(
                    n_components=10, weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=1e-3, max_iter=200, random_state=self.seed,
                ).fit(source.reshape(-1, 1))
                keep = gmm.weights_ > 5e-3
                info["gmm"] = gmm
                info["modes"] = np.flatnonzero(keep)
                self.spans.append(("num", 1, column))
                self.spans.append(("cat", int(keep.sum()), column + "__mode"))
            else:
                self.spans.append(("num", 1, column))
            self.info[column] = info
        self.levels = {}
        for column in CATEGORICAL:
            levels = sorted(frame[column].astype(str).unique())
            self.levels[column] = levels
            self.spans.append(("cat", len(levels), column))
        self.width = sum(width for _kind, width, _name in self.spans)
        self.cat_spans = [(i, w, n) for i, (k, w, n) in enumerate(self.spans) if k == "cat" and n in CATEGORICAL]
        return self

    def transform(self, frame):
        blocks = []
        for kind, width, name in self.spans:
            column = name.split("__")[0]
            if column in NUMERIC:
                info = self.info[column]
                values = frame[column].to_numpy(dtype=float)
                if name.endswith("__zero"):
                    blocks.append(np.eye(2)[(values > 0).astype(int)])
                    continue
                source = values
                if info.get("zero"):
                    source = np.log1p(np.maximum(values, 0.0))
                elif info.get("log"):
                    source = np.log1p(values)
                if "gmm" in info:
                    # Filled by _align_modes so the alpha value and mode one-hot share one sampled mode.
                    blocks.append(np.zeros((len(frame), width)))
                    continue
                low, high = (info["lmin"], info["lmax"]) if info.get("zero") else (info["min"], info["max"])
                blocks.append(((source - low) / max(high - low, 1e-9)).clip(0, 1).reshape(-1, 1))
            else:
                observed = frame[column].astype(str).to_numpy()
                blocks.append(np.stack([(observed == level).astype(float) for level in self.levels[column]], 1))
        # Mode one-hots must use the same sampled mode as the alpha value.
        out = np.hstack(blocks)
        if self.config["norm"] == "gmm":
            out = self._align_modes(frame, out)
        return out.astype(np.float32)

    def _align_modes(self, frame, out):
        start = 0
        rng = np.random.default_rng(self.seed + 1)
        for kind, width, name in self.spans:
            column = name.split("__")[0]
            if kind == "num" and column in NUMERIC and "gmm" in self.info[column]:
                info = self.info[column]
                values = frame[column].to_numpy(dtype=float)
                source = np.log1p(np.maximum(values, 0.0)) if (info.get("zero") or info.get("log")) else values
                gmm, modes = info["gmm"], info["modes"]
                means = gmm.means_.reshape(-1)[modes]
                stds = np.sqrt(gmm.covariances_.reshape(-1))[modes]
                probs = gmm.predict_proba(source.reshape(-1, 1))[:, modes] + 1e-6
                probs = probs / probs.sum(axis=1, keepdims=True)
                cumulative = probs.cumsum(axis=1)
                chosen = (rng.random((len(probs), 1)) < cumulative).argmax(axis=1)
                out[:, start] = np.clip((source - means[chosen]) / (4 * stds[chosen]), -0.99, 0.99)
                width_next = len(modes)
                out[:, start + 1:start + 1 + width_next] = np.eye(width_next)[chosen]
            start += width
        return out

    def inverse(self, matrix, rng):
        """Encoded rows to an original-domain frame."""
        data, start, pending = {}, 0, {}
        for kind, width, name in self.spans:
            block = matrix[:, start:start + width]
            start += width
            column = name.split("__")[0]
            if name.endswith("__zero"):
                pending[column + "_nonzero"] = self._pick(block, rng) == 1
            elif name.endswith("__mode"):
                pending[column + "_mode"] = self._pick(block, rng)
            elif kind == "num":
                pending[column + "_raw"] = block[:, 0]
            else:
                data[column] = np.array(self.levels[column])[self._pick(block, rng)]
        for column in NUMERIC:
            info = self.info[column]
            raw = pending[column + "_raw"]
            if "gmm" in info:
                gmm, modes = info["gmm"], info["modes"]
                means = gmm.means_.reshape(-1)[modes]
                stds = np.sqrt(gmm.covariances_.reshape(-1))[modes]
                mode = pending[column + "_mode"]
                source = np.clip(raw, -1, 1) * 4 * stds[mode] + means[mode]
            else:
                low, high = (info["lmin"], info["lmax"]) if info.get("zero") else (info["min"], info["max"])
                source = low + raw * (high - low)
            values = np.expm1(source) if (info.get("zero") or info.get("log")) else source
            if info.get("zero"):
                values = np.where(pending[column + "_nonzero"], values, 0.0)
            values = np.clip(values, info["min"], info["max"])
            if self.config["round_int"] and column in INTEGER:
                values = np.round(values)
            data[column] = values
        return pd.DataFrame(data)[NUMERIC + CATEGORICAL]

    def _pick(self, block, rng):
        if self.config["cat_sample"] == "argmax" or block.shape[1] == 1:
            return block.argmax(axis=1)
        probs = np.clip(block, 1e-8, None)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return (rng.random((len(probs), 1)) < probs.cumsum(axis=1)).argmax(axis=1)


# --------------------------------------------------------------- model ----

class Generator(nn.Module):
    def __init__(self, transformer, config, cond_dim):
        super().__init__()
        self.spans = transformer.spans
        self.config = config
        hidden = config["hidden_dim"]
        self.net = nn.Sequential(
            nn.Linear(config["latent_dim"] + cond_dim, hidden), nn.BatchNorm1d(hidden) if config["loss"] == "wgangp" else nn.Identity(), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden) if config["loss"] == "wgangp" else nn.Identity(), nn.ReLU(),
            nn.Linear(hidden, transformer.width),
        )

    def forward(self, noise, cond, temperature):
        raw = self.net(torch.cat([noise, cond], 1))
        if self.config["cat_train"] == "sigmoid":
            return torch.sigmoid(raw), raw
        parts, start = [], 0
        for kind, width, name in self.spans:
            chunk = raw[:, start:start + width]
            start += width
            if kind == "num":
                parts.append(torch.tanh(chunk) if self.config["norm"] == "gmm" and name.split("__")[0] in NUMERIC else torch.sigmoid(chunk))
            else:
                hard = self.config["cat_train"] == "gumbel_st"
                parts.append(F.gumbel_softmax(chunk, tau=temperature, hard=hard))
        return torch.cat(parts, 1), raw

    @torch.no_grad()
    def probabilities(self, noise, cond):
        """Per-span probabilities (softmax) for sampling at generation time."""
        raw = self.net(torch.cat([noise, cond], 1))
        if self.config["cat_train"] == "sigmoid":
            return torch.sigmoid(raw)
        parts, start = [], 0
        for kind, width, name in self.spans:
            chunk = raw[:, start:start + width]
            start += width
            if kind == "num":
                parts.append(torch.tanh(chunk) if self.config["norm"] == "gmm" and name.split("__")[0] in NUMERIC else torch.sigmoid(chunk))
            else:
                parts.append(torch.softmax(chunk, 1))
        return torch.cat(parts, 1)


class Discriminator(nn.Module):
    def __init__(self, width, cond_dim, config):
        super().__init__()
        self.pac = config["pac"]
        hidden = config["hidden_dim"]
        layers = [nn.Linear((width + cond_dim) * self.pac, hidden), nn.LeakyReLU(0.2),
                  nn.Linear(hidden, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, 1)]
        if config["loss"] == "wgangp":
            layers.insert(2, nn.Dropout(0.5))
            layers.insert(5, nn.Dropout(0.5))
        self.net = nn.Sequential(*layers)

    def forward(self, rows, cond):
        joined = torch.cat([rows, cond], 1)
        return self.net(joined.view(-1, joined.shape[1] * self.pac))


class Conditioner:
    """Income one-hot, plus a training-by-sampling categorical condition when enabled."""

    def __init__(self, transformer, encoded, income, config, seed):
        self.config = config
        self.income = income
        self.rng = np.random.default_rng(seed)
        self.base_rate = float(income.mean())
        self.tbs = config["tbs"]
        self.encoded = encoded
        self.offsets, self.cat_dim, start = [], 0, 0
        span_starts = []
        for kind, width, name in transformer.spans:
            span_starts.append(start)
            start += width
        if self.tbs:
            for index, width, name in transformer.cat_spans:
                block = encoded[:, span_starts[index]:span_starts[index] + width]
                category = block.argmax(1)
                freq = np.log1p(np.bincount(category, minlength=width).astype(float))
                rows_by_cat = [np.flatnonzero(category == k) for k in range(width)]
                self.offsets.append({"col_start": span_starts[index], "width": width,
                                     "cond_start": self.cat_dim, "p": freq / freq.sum(),
                                     "rows": rows_by_cat})
                self.cat_dim += width
        self.dim = 2 + self.cat_dim

    def sample_train(self, n):
        """Conditions and matching real row indexes for one training batch."""
        cond = np.zeros((n, self.dim), dtype=np.float32)
        if self.tbs:
            picks = self.rng.integers(len(self.offsets), size=n)
            real_rows = np.empty(n, dtype=int)
            masks = []
            for i, pick in enumerate(picks):
                info = self.offsets[pick]
                k = self.rng.choice(info["width"], p=info["p"])
                while len(info["rows"][k]) == 0:
                    k = self.rng.choice(info["width"], p=info["p"])
                cond[i, 2 + info["cond_start"] + k] = 1.0
                real_rows[i] = self.rng.choice(info["rows"][k])
                masks.append((info["col_start"], info["width"], k))
            cond[np.arange(n), self.income[real_rows]] = 1.0
            return cond, real_rows, masks
        real_rows = self.rng.integers(len(self.income), size=n)
        if self.config["label_prior"] == "empirical":
            labels = (self.rng.random(n) < self.base_rate).astype(int)
        else:
            labels = self.rng.integers(2, size=n)
        cond[np.arange(n), labels] = 1.0
        real_cond = np.zeros_like(cond)
        real_cond[np.arange(n), self.income[real_rows]] = 1.0
        return (cond, real_cond), real_rows, None

    def for_labels(self, labels, seed=0):
        labels = np.asarray(labels, dtype=int)
        cond = np.zeros((len(labels), self.dim), dtype=np.float32)
        cond[np.arange(len(labels)), labels] = 1.0
        if self.tbs:
            # Condition on the category of a real fit row with the same income, as CTGAN
            # samples conditions from the data distribution at generation time.
            rng = np.random.default_rng(seed)
            for i, label in enumerate(labels):
                row = rng.choice(np.flatnonzero(self.income == label))
                info = self.offsets[rng.integers(len(self.offsets))]
                k = int(self.encoded[row, info["col_start"]:info["col_start"] + info["width"]].argmax())
                cond[i, 2 + info["cond_start"] + k] = 1.0
        return cond


def gradient_penalty(critic, real, fake, cond, pac):
    alpha = torch.rand(real.shape[0] // pac, 1, 1).repeat(1, pac, real.shape[1]).view(-1, real.shape[1])
    mixed = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    out = critic(mixed, cond)
    grad = torch.autograd.grad(out.sum(), mixed, create_graph=True)[0]
    grad = grad.reshape(-1, pac * real.shape[1])
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


def rule_penalty(raw, transformer):
    """Expected number of broken rules per row, from the generator's category probabilities."""
    probs, start = {}, 0
    for kind, width, name in transformer.spans:
        if kind == "cat" and name in CATEGORICAL:
            probs[name] = torch.softmax(raw[:, start:start + width], 1)
        start += width
    return expected_violations(probs, transformer.levels).mean()


def expected_violations(probs, levels):
    """Expected number of broken domain rules per row, given category probabilities per column.

    probs maps each categorical column to an (n, levels) tensor of probabilities. Shared by the
    GAN (step 5) and the diffusion steps, so every model is held to the same rules.
    """
    level = lambda column, value: probs[column][:, levels[column].index(value)]
    husband, wife = level("relationship", "Husband"), level("relationship", "Wife")
    married = sum(level("marital-status", m) for m in MARRIED if m in levels["marital-status"])
    no_job = sum(level("workclass", w) for w in NO_OCCUPATION if w in levels["workclass"])
    occ_missing = level("occupation", "?")
    penalty = husband * level("sex", "Female") + wife * level("sex", "Male")
    penalty = penalty + (husband + wife) * (1 - married)
    penalty = penalty + occ_missing * (1 - no_job) + (1 - occ_missing) * no_job
    return penalty


if DATASET == "acs":
    expected_violations = acs_data.expected_violations  # noqa: F811


# ------------------------------------------------------------ training ----

def _draw(generator, transformer, conditioner, labels, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    generator.eval()
    cond = torch.tensor(conditioner.for_labels(labels, seed))
    noise = torch.randn(len(labels), generator.config["latent_dim"])
    probs = generator.probabilities(noise, cond).numpy()
    generator.train()
    return transformer.inverse(probs, rng)


def generate(generator, transformer, conditioner, labels, seed, reject=None, stats=None):
    """Synthetic rows for the given income labels.

    With rejection on (the rung's config, or reject=True), rows that break a
    domain rule are redrawn for the same label, up to 50 rounds. The share of
    redrawn rows is written to stats["rejected_share"] when stats is a dict.
    """
    labels = np.asarray(labels, dtype=int)
    frame = _draw(generator, transformer, conditioner, labels, seed)
    reject = generator.config.get("reject", False) if reject is None else reject
    if not reject:
        return frame
    bad, _counts = rule_violations(frame)
    first_bad = int(bad.sum())
    for round_ in range(1, 51):
        if not bad.any():
            break
        index = np.flatnonzero(bad)
        redraw = _draw(generator, transformer, conditioner, labels[index], seed + 100003 * round_)
        frame.iloc[index] = redraw.to_numpy()
        bad[index] = rule_violations(redraw)[0]
    if stats is not None:
        stats["rejected_share"] = first_bad / max(len(frame), 1)
        stats["still_broken"] = int(bad.sum())
    return frame


def score(real_frame, synth_frame, schema):
    validity = hard_validity(synth_frame, schema)
    distances = fidelity_report(real_frame, synth_frame, schema)
    detection = selection_detection(real_frame, synth_frame, schema)
    return {
        "violations": int(validity["violation_total"]),
        "ks": distances["numeric_ks_mean"], "tv": distances["categorical_tv_mean"],
        "coverage": distances["category_coverage_mean"], "rare_recall": distances["rare_category_recall"],
        "correlation_mae": distances["correlation_mae"], "duplicates": distances["synthetic_duplicate_rate"],
        "selection_auc": detection["mean_auc"],
    }


def _snapshot_epochs(epochs):
    """Four evenly spaced epochs kept for the reverse-engineering study (memorization over training)."""
    return sorted({max(1, round(epochs * q)) for q in (0.25, 0.5, 0.75, 1.0)})


def _clone(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def train_rung(rung, config, seed, fit, select, test, schema, log, model_dir=None, split=None, resume_path=None):
    """resume_path: optional file where the full training state is saved after every scoring epoch.
    If it exists, training continues from it, with every random number generator restored, so the
    result is the same as an uninterrupted run (tests/test_acs.py checks this)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    transformer = Transformer(config, seed).fit(fit)
    encoded = transformer.transform(fit)
    income = fit["income"].to_numpy()
    conditioner = Conditioner(transformer, encoded, income, config, seed)
    generator = Generator(transformer, config, conditioner.dim)
    critic = Discriminator(transformer.width, conditioner.dim, config)
    opt_g = torch.optim.Adam(generator.parameters(), lr=config["lr"], betas=config["betas"])
    opt_d = torch.optim.Adam(critic.parameters(), lr=config["lr"], betas=config["betas"])
    data = torch.tensor(encoded)
    batch = config["batch_size"]
    steps = max(len(encoded) // batch, 1)
    bce = nn.BCEWithLogitsLoss()
    history, best = [], None
    losses, snapshots, keep = [], {}, _snapshot_epochs(config["epochs"])
    select_features = select.drop(columns=["income"])
    started, first_epoch = time.time(), 1
    if resume_path is not None and Path(resume_path).exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        generator.load_state_dict(state["generator"])
        critic.load_state_dict(state["critic"])
        opt_g.load_state_dict(state["opt_g"])
        opt_d.load_state_dict(state["opt_d"])
        conditioner.rng.bit_generator.state = state["conditioner_rng"]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        history, best, losses, snapshots = state["history"], state["best"], state["losses"], state["snapshots"]
        first_epoch, started = state["epoch"] + 1, time.time() - state["elapsed"]
        log(f"{rung} seed {seed}: resumed after epoch {state['epoch']}")
    for epoch in range(first_epoch, config["epochs"] + 1):
        sum_d = sum_g = 0.0
        for _step in range(steps):
            for _ in range(config["critic_steps"]):
                conds, rows, masks = conditioner.sample_train(batch)
                fake_cond = torch.tensor(conds if config["tbs"] else conds[0])
                real_cond = torch.tensor(conds if config["tbs"] else conds[1])
                real = data[rows]
                noise = torch.randn(batch, config["latent_dim"])
                fake, _raw = generator(noise, fake_cond, config["temperature"])
                d_real = critic(real, real_cond)
                d_fake = critic(fake.detach(), fake_cond)
                if config["loss"] == "wgangp":
                    loss_d = d_fake.mean() - d_real.mean()
                    loss_d = loss_d + config["gp_weight"] * gradient_penalty(critic, real, fake.detach(), real_cond, config["pac"])
                else:
                    loss_d = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()
                sum_d += float(loss_d.detach())
            conds, rows, masks = conditioner.sample_train(batch)
            fake_cond = torch.tensor(conds if config["tbs"] else conds[0])
            noise = torch.randn(batch, config["latent_dim"])
            fake, raw = generator(noise, fake_cond, config["temperature"])
            d_fake = critic(fake, fake_cond)
            loss_g = -d_fake.mean() if config["loss"] == "wgangp" else bce(d_fake, torch.ones_like(d_fake))
            if config.get("rule_weight"):
                loss_g = loss_g + config["rule_weight"] * rule_penalty(raw, transformer)
            if config["tbs"]:
                targets = torch.tensor([k for _s, _w, k in masks])
                ce = torch.stack([F.cross_entropy(raw[i:i + 1, s:s + w], targets[i:i + 1])
                                  for i, (s, w, _k) in enumerate(masks)]).mean()
                loss_g = loss_g + ce
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()
            sum_g += float(loss_g.detach())
        # Loss per epoch and a few weight snapshots; neither touches a random number generator.
        losses.append({"epoch": epoch, "loss_d": sum_d / (steps * config["critic_steps"]), "loss_g": sum_g / steps})
        if epoch in keep:
            snapshots[epoch] = {"generator_state": _clone(generator), "critic_state": _clone(critic)}
        if epoch % config["score_every"] == 0 or epoch == config["epochs"]:
            synth = generate(generator, transformer, conditioner, select["income"].to_numpy(), seed=42)
            row = {"epoch": epoch, "seconds": round(time.time() - started, 1), **score(select_features, synth, schema)}
            history.append(row)
            log(f"{rung} seed {seed} epoch {epoch}: auc {row['selection_auc']:.4f} ks {row['ks']:.3f} "
                f"tv {row['tv']:.3f} cov {row['coverage']:.3f} viol {row['violations']} ({row['seconds']}s)")
            # Predeclared rule: fewest violations, then lowest selection AUC, then earlier epoch.
            key = (row["violations"] > 0, row["selection_auc"], epoch)
            if best is None or key < best["key"]:
                best = {"key": key, "row": row, "state": _clone(generator), "critic_state": _clone(critic)}
            if resume_path is not None:
                tmp = Path(str(resume_path) + ".tmp")
                torch.save({"epoch": epoch, "elapsed": time.time() - started,
                            "generator": generator.state_dict(), "critic": critic.state_dict(),
                            "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                            "conditioner_rng": conditioner.rng.bit_generator.state,
                            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                            "history": history, "best": best, "losses": losses, "snapshots": snapshots}, tmp)
                tmp.replace(resume_path)
    final_state = {"generator_state": _clone(generator), "critic_state": _clone(critic),
                   "opt_g_state": opt_g.state_dict(), "opt_d_state": opt_d.state_dict(), "epoch": config["epochs"]}
    generator.load_state_dict(best["state"])
    model_dir = Path(model_dir or MODEL_DIR)
    if test is None:
        # No test rows during training (ACS protocol): scoring happens once, later, in acs.py.
        _save_rung(model_dir, rung, seed, config, best, generator, transformer, conditioner, final_state,
                   snapshots, losses, history, split)
        return {"rung": rung, "seed": seed,
                "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in config.items()},
                "selected": best["row"], "history": history, "train_seconds": round(time.time() - started, 1)}
    # Final scoring, once, after selection.
    dev_synth = generate(generator, transformer, conditioner, select["income"].to_numpy(), seed=7)
    test_synth = generate(generator, transformer, conditioner, test["income"].to_numpy(), seed=9000 + seed)
    final = final_detection(select_features, dev_synth, test.drop(columns=["income"]), test_synth, schema)
    fit_labels = fit["income"].to_numpy()
    utility_synth = generate(generator, transformer, conditioner, fit_labels, seed=11)
    utility = utility_report(
        _design_matrix(fit.drop(columns=["income"]), schema), fit_labels,
        _design_matrix(utility_synth, schema), fit_labels,
        _design_matrix(test.drop(columns=["income"]), schema), test["income"].to_numpy(),
    )
    test_fidelity = score(test.drop(columns=["income"]), test_synth, schema)
    _save_rung(model_dir, rung, seed, config, best, generator, transformer, conditioner, final_state,
               snapshots, losses, history, split)
    return {
        "rung": rung, "seed": seed, "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in config.items()},
        "selected": best["row"], "history": history,
        "final_detection_auc": final["mean_auc"], "utility_ratio": utility["ratio"],
        "utility_auc_real": utility["auc_real"], "utility_auc_synthetic": utility["auc_synthetic"],
        "test_ks": test_fidelity["ks"], "test_tv": test_fidelity["tv"], "test_coverage": test_fidelity["coverage"],
        "test_violations": test_fidelity["violations"],
    }


def _save_rung(model_dir, rung, seed, config, best, generator, transformer, conditioner, final_state,
               snapshots, losses, history, split):
    model_dir.mkdir(parents=True, exist_ok=True)
    # Everything the reverse-engineering study needs, so it can run without retraining:
    # the critic at the selected epoch, final weights and optimizer states (to resume),
    # snapshots over training, the loss curve, and which rows were members of the fit slice.
    torch.save({
        "rung": rung, "seed": seed, "config": config, "selected_epoch": best["row"]["epoch"],
        "generator_state": generator.state_dict(), "critic_state": best["critic_state"],
        "transformer": transformer,
        "conditioner": {"dim": conditioner.dim, "tbs": conditioner.tbs, "offsets": conditioner.offsets,
                        "income": conditioner.income, "encoded": conditioner.encoded,
                        "base_rate": conditioner.base_rate},
        "final": final_state, "snapshots": snapshots, "losses": losses, "history": history,
        "split": split, "torch_version": torch.__version__,
    }, model_dir / f"{rung}_seed{seed}.pt")


def load_generator(path, reject=None):
    """Rebuild a saved generator. Returns a function: income labels -> synthetic frame.

    reject=None keeps the rung's own setting; True or False overrides it.
    """
    saved = torch.load(path, map_location="cpu", weights_only=False)
    conditioner = Conditioner.__new__(Conditioner)
    conditioner.__dict__.update(saved["conditioner"])
    conditioner.config = saved["config"]
    generator = Generator(saved["transformer"], saved["config"], conditioner.dim)
    generator.load_state_dict(saved["generator_state"])

    def sample(labels, seed=0, stats=None):
        frame = generate(generator, saved["transformer"], conditioner, np.asarray(labels, dtype=int), seed,
                         reject=reject, stats=stats)
        frame["income"] = np.asarray(labels, dtype=int)
        return frame

    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rungs", nargs="+", default=list(RUNGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=None, help="override for smoke tests")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="where result JSON files go")
    parser.add_argument("--model-dir", default=str(MODEL_DIR), help="where checkpoints go")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = load_adult()
    schema = schema_from_frame(frame, NUMERIC, CATEGORICAL)
    torch.set_num_threads(int(__import__("os").environ.get("LADDER_THREADS", "1")))
    for rung in args.rungs:
        for seed in args.seeds:
            out = out_dir / f"{rung}_seed{seed}.json"
            if out.exists() and args.epochs is None:
                print("skip", out.name, flush=True)
                continue
            config = dict(RUNGS[rung])
            if args.epochs:
                config["epochs"] = args.epochs
                config["score_every"] = args.epochs
            fit, select, test = split_frames(frame, seed)
            result = train_rung(rung, config, seed, fit, select, test, schema, lambda m: print(m, flush=True),
                                model_dir=args.model_dir, split=split_indices(frame, seed))
            if args.epochs is None:
                out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
            print(f"DONE {rung} seed {seed}: final AUC {result['final_detection_auc']:.4f} "
                  f"utility {result['utility_ratio']:.3f}", flush=True)


if __name__ == "__main__":
    # Run through the importable module so saved objects unpickle as ladder.Transformer.
    import ladder

    ladder.main()

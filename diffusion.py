"""Diffusion steps of the ladder (7-9), trained and scored with the same protocol as steps 0-5.

  7 ddpm          Gaussian diffusion on the numeric columns and multinomial diffusion on the
                  categorical columns, in the style of TabDDPM, conditioned on income. No rules.
  8 ddpm_rules    step 7 plus the domain-rule penalty on the model's predicted categories,
                  and the rule filter at sampling time
  9 latent_rules  latent diffusion in the style of TabSyn: a small transformer VAE turns each
                  row into a continuous vector, and an EDM diffusion model learns those vectors,
                  conditioned on income. Same rule penalty (through the frozen decoder) and filter.

Step 10, step 5's GAN rows refined by a few denoising steps of step 8, uses no training and
lives in evaluate_holdout.py.

Data, splits, selection rule and scoring are the ones in ladder.py: models are fit on the fit
slice only, the checkpoint is chosen on the selection slice only (zero hard-validity
violations first, then the lowest detection AUC), and the internal test split is scored once.

Each checkpoint also keeps what the reverse-engineering study needs, so it can run without
retraining: final weights and optimizer state (to resume), the EMA weights, weight snapshots
over training, the loss curve, the noise schedule, the encoders and which rows were members
of the fit slice. `per_row_loss` gives the denoising error of each row at a chosen noise level,
which is the white-box membership score for diffusion models.

Importing this module does not read data or train.
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer

from evaluator import _design_matrix, final_detection, schema_from_frame, utility_report
from ladder import (
    CATEGORICAL,
    INTEGER,
    MODEL_DIR,
    NUMERIC,
    OUT_DIR,
    SEEDS,
    ZERO_INFLATED,
    expected_violations,
    rule_violations,
    score,
    split_frames,
    split_indices,
)
from main import load_adult

STEPS = {
    "7_ddpm": {
        "kind": "ddpm", "timesteps": 1000, "hidden": (512, 512, 512, 512), "dim_t": 128,
        "batch_size": 1024, "lr": 1e-3, "weight_decay": 0.0, "train_steps": 30000, "ema": 0.999,
        "sample_steps": 100, "score_every": 3000, "rule_weight": 0.0, "reject": False,
    },
}
STEPS["8_ddpm_rules"] = dict(STEPS["7_ddpm"], rule_weight=1.0, reject=True)
STEPS["9_latent_rules"] = {
    "kind": "latent", "d_token": 4, "vae_layers": 2, "vae_ff": 128, "vae_epochs": 300,
    "vae_batch": 1024, "vae_lr": 1e-3, "beta_max": 1e-2, "beta_min": 1e-5, "beta_decay": 0.7,
    "beta_patience": 10, "hidden": (512, 512, 512, 512), "dim_t": 128, "batch_size": 1024,
    "lr": 1e-3, "weight_decay": 0.0, "train_steps": 30000, "ema": 0.999, "sample_steps": 50,
    "sigma_data": 0.5, "p_mean": -1.2, "p_std": 1.2, "sigma_min": 0.002, "sigma_max": 80.0,
    "rho": 7.0, "score_every": 3000, "rule_weight": 1.0, "reject": True,
}


# ------------------------------------------------------------- encoder ----

class TabEncoder:
    """Numeric columns to a normal scale with a quantile transform, categories to indexes.

    capital-gain and capital-loss get a zero / non-zero gate, kept as an extra categorical
    column, and the quantile transform is fit on the non-zero values only (the lesson of
    step 1: a quantile map cannot represent a spike of exact zeros). Fit on the fit slice only.
    """

    def __init__(self, seed):
        self.seed = seed

    def fit(self, frame):
        self.quantile, self.bounds = {}, {}
        for column in NUMERIC:
            values = frame[column].to_numpy(dtype=float)
            self.bounds[column] = (float(values.min()), float(values.max()))
            if column in ZERO_INFLATED:
                values = np.log1p(values[values > 0])
            qt = QuantileTransformer(n_quantiles=min(1000, len(values)), output_distribution="normal",
                                     subsample=10 ** 9, random_state=self.seed)
            self.quantile[column] = qt.fit(values.reshape(-1, 1))
        self.cat_columns = [c + "__zero" for c in NUMERIC if c in ZERO_INFLATED] + list(CATEGORICAL)
        self.levels = {c + "__zero": ["0", "1"] for c in NUMERIC if c in ZERO_INFLATED}
        for column in CATEGORICAL:
            self.levels[column] = sorted(frame[column].astype(str).unique())
        self.num_classes = [len(self.levels[c]) for c in self.cat_columns]
        return self

    def transform(self, frame):
        x_num = np.zeros((len(frame), len(NUMERIC)), dtype=np.float32)
        for j, column in enumerate(NUMERIC):
            values = frame[column].to_numpy(dtype=float)
            if column in ZERO_INFLATED:
                positive = values > 0
                if positive.any():
                    x_num[positive, j] = self.quantile[column].transform(np.log1p(values[positive]).reshape(-1, 1))[:, 0]
            else:
                x_num[:, j] = self.quantile[column].transform(values.reshape(-1, 1))[:, 0]
        x_cat = np.zeros((len(frame), len(self.cat_columns)), dtype=np.int64)
        for j, column in enumerate(self.cat_columns):
            if column.endswith("__zero"):
                x_cat[:, j] = (frame[column[:-6]].to_numpy(dtype=float) > 0).astype(int)
            else:
                lookup = {level: k for k, level in enumerate(self.levels[column])}
                # A level never seen in the fit slice maps to the first level (cannot happen for fit rows).
                x_cat[:, j] = [lookup.get(v, 0) for v in frame[column].astype(str)]
        return x_num, x_cat

    def inverse(self, x_num, x_cat):
        data = {}
        gates = {c[:-6]: x_cat[:, j] for j, c in enumerate(self.cat_columns) if c.endswith("__zero")}
        for j, column in enumerate(NUMERIC):
            values = self.quantile[column].inverse_transform(np.asarray(x_num[:, j:j + 1], dtype=float))[:, 0]
            if column in ZERO_INFLATED:
                values = np.where(gates[column] == 1, np.expm1(values), 0.0)
            low, high = self.bounds[column]
            values = np.clip(values, low, high)
            if column in INTEGER:
                values = np.round(values)
            data[column] = values
        for j, column in enumerate(self.cat_columns):
            if not column.endswith("__zero"):
                data[column] = np.array(self.levels[column])[x_cat[:, j]]
        return pd.DataFrame(data)[NUMERIC + CATEGORICAL]

    def cat_probs(self, probs_list):
        """Dict of the real categorical columns' probabilities, for the rule penalty."""
        return {c: p for c, p in zip(self.cat_columns, probs_list) if not c.endswith("__zero")}


# ------------------------------------------------------------ networks ----

def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)


class Denoiser(nn.Module):
    """TabDDPM-style MLP: projected input + time embedding + income embedding, then an MLP."""

    def __init__(self, d_in, d_out, hidden, dim_t, n_labels=2):
        super().__init__()
        self.dim_t = dim_t
        self.proj = nn.Linear(d_in, dim_t)
        self.time = nn.Sequential(nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t))
        self.label = nn.Embedding(n_labels, dim_t)
        layers, width = [], dim_t
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        layers.append(nn.Linear(width, d_out))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, t_embed_input, y):
        h = self.proj(x) + self.time(timestep_embedding(t_embed_input, self.dim_t)) + self.label(y)
        return self.mlp(h)


# --------------------------------------------------- step 7-8: TabDDPM ----

def cosine_alphas_cumprod(timesteps, s=0.008):
    """alpha-bar for t = 0..T (alpha-bar_0 = 1), cosine schedule as in improved DDPM and TabDDPM."""
    steps = np.arange(timesteps + 1, dtype=np.float64) / timesteps
    f = np.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = np.clip(1 - alpha_bar[1:] / alpha_bar[:-1], 0, 0.999)
    return np.concatenate([[1.0], np.cumprod(1 - betas)])


class GaussianMultinomialDiffusion(nn.Module):
    """Gaussian diffusion for numeric columns, multinomial diffusion for categorical columns.

    Time t runs over 1..T; t = 0 is clean data. The network predicts the Gaussian noise and
    the clean categories (x0 logits). Sampling may skip steps: both processes compose, so the
    posterior from step t to any earlier step s has a closed form.
    """

    def __init__(self, n_num, num_classes, config):
        super().__init__()
        self.n_num, self.num_classes = n_num, list(num_classes)
        self.T = config["timesteps"]
        self.register_buffer("alpha_bar", torch.tensor(cosine_alphas_cumprod(self.T), dtype=torch.float32))
        self.offsets = np.concatenate([[0], np.cumsum(self.num_classes)]).tolist()
        width = n_num + sum(self.num_classes)
        self.net = Denoiser(width, width, config["hidden"], config["dim_t"])

    # -- helpers
    def _onehot(self, x_cat):
        return torch.cat([F.one_hot(x_cat[:, j], k).float() for j, k in enumerate(self.num_classes)], 1)

    def _split_logits(self, out):
        eps = out[:, :self.n_num]
        logits = [out[:, self.n_num + a:self.n_num + b] for a, b in zip(self.offsets[:-1], self.offsets[1:])]
        return eps, logits

    def predict(self, x_num_t, x_cat_t, t, y):
        out = self.net(torch.cat([x_num_t, self._onehot(x_cat_t)], 1), t, y)
        return self._split_logits(out)

    def _cat_posterior(self, x_t_onehot, x0_probs, ab_t, ab_s, k):
        """q(x_s | x_t, x0) for one categorical column, s < t (ab_* are (n, 1) tensors)."""
        alpha_ts = ab_t / ab_s
        theta = (alpha_ts * x_t_onehot + (1 - alpha_ts) / k) * (ab_s * x0_probs + (1 - ab_s) / k)
        return theta / theta.sum(1, keepdim=True)

    def q_sample(self, x_num, x_cat, t, generator=None):
        ab = self.alpha_bar[t][:, None]
        noise = torch.randn(x_num.shape, generator=generator)
        x_num_t = ab.sqrt() * x_num + (1 - ab).sqrt() * noise
        cats = []
        for j, k in enumerate(self.num_classes):
            probs = ab * F.one_hot(x_cat[:, j], k).float() + (1 - ab) / k
            cats.append(torch.multinomial(probs, 1, generator=generator)[:, 0])
        return x_num_t, torch.stack(cats, 1), noise

    def per_row_loss(self, x_num, x_cat, y, t, generator=None, return_probs=False):
        """Denoising loss of each row at noise level t (TabDDPM's loss, not averaged over rows)."""
        x_num_t, x_cat_t, noise = self.q_sample(x_num, x_cat, t, generator)
        eps, logits = self.predict(x_num_t, x_cat_t, t, y)
        gauss = ((eps - noise) ** 2).mean(1)
        ab_t = self.alpha_bar[t][:, None]
        ab_s = self.alpha_bar[t - 1][:, None]
        kl, probs_list = 0.0, []
        for j, k in enumerate(self.num_classes):
            x_t_oh = F.one_hot(x_cat_t[:, j], k).float()
            probs = torch.softmax(logits[j], 1)
            probs_list.append(probs)
            true_post = self._cat_posterior(x_t_oh, F.one_hot(x_cat[:, j], k).float(), ab_t, ab_s, k)
            model_post = self._cat_posterior(x_t_oh, probs, ab_t, ab_s, k)
            kl = kl + (true_post * (torch.log(true_post.clamp_min(1e-30)) - torch.log(model_post.clamp_min(1e-30)))).sum(1)
        multi = kl / len(self.num_classes)
        if return_probs:
            return gauss, multi, probs_list
        return gauss + multi

    @torch.no_grad()
    def sample(self, y, steps, generator=None, start=None, t_start=None):
        """Ancestral sampling over `steps` evenly spaced steps from t_start (default T) to 0.

        With `start` = (x_num, x_cat) already noised to t_start, this continues denoising
        from there (used by the step 10 refinement).
        """
        n = len(y)
        t_start = self.T if t_start is None else int(t_start)
        if start is None:
            x_num = torch.randn((n, self.n_num), generator=generator)
            x_cat = torch.stack([torch.randint(k, (n,), generator=generator) for k in self.num_classes], 1)
        else:
            x_num, x_cat = start
        grid = np.unique(np.linspace(0, t_start, steps + 1).round().astype(int))[::-1]
        for t_value, s_value in zip(grid[:-1], grid[1:]):
            t = torch.full((n,), int(t_value), dtype=torch.long)
            eps, logits = self.predict(x_num, x_cat, t, y)
            ab_t = self.alpha_bar[int(t_value)]
            ab_s = self.alpha_bar[int(s_value)]
            x0 = ((x_num - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-6, 6)
            if s_value == 0:
                x_num = x0
            else:
                alpha_ts = ab_t / ab_s
                beta_ts = 1 - alpha_ts
                mean = (ab_s.sqrt() * beta_ts / (1 - ab_t)) * x0 + (alpha_ts.sqrt() * (1 - ab_s) / (1 - ab_t)) * x_num
                var = beta_ts * (1 - ab_s) / (1 - ab_t)
                x_num = mean + var.sqrt() * torch.randn(x_num.shape, generator=generator)
            cats = []
            for j, k in enumerate(self.num_classes):
                probs = torch.softmax(logits[j], 1)
                post = self._cat_posterior(F.one_hot(x_cat[:, j], k).float(), probs,
                                           ab_t.expand(n, 1), ab_s.expand(n, 1), k)
                cats.append(torch.multinomial(post, 1, generator=generator)[:, 0])
            x_cat = torch.stack(cats, 1)
        return x_num, x_cat


# ------------------------------------------------- step 9: latent (TabSyn) ----

class Tokenizer(nn.Module):
    def __init__(self, n_num, num_classes, d):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_num, d) / math.sqrt(d))
        self.bias = nn.Parameter(torch.zeros(n_num + len(num_classes), d))
        self.embeds = nn.ModuleList([nn.Embedding(k, d) for k in num_classes])

    def forward(self, x_num, x_cat):
        num = x_num[:, :, None] * self.weight[None]
        cat = torch.stack([emb(x_cat[:, j]) for j, emb in enumerate(self.embeds)], 1)
        return torch.cat([num, cat], 1) + self.bias[None]


class VAE(nn.Module):
    """Column tokens -> transformer -> one d-dim latent token per column, and back (TabSyn's design)."""

    def __init__(self, n_num, num_classes, config):
        super().__init__()
        d, layers, ff = config["d_token"], config["vae_layers"], config["vae_ff"]
        make = lambda: nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, 1, ff, dropout=0.0, batch_first=True), layers, enable_nested_tensor=False)
        self.n_num, self.num_classes = n_num, list(num_classes)
        self.tokenizer = Tokenizer(n_num, num_classes, d)
        self.enc_mu, self.enc_logvar, self.dec = make(), make(), make()
        self.num_out = nn.ModuleList([nn.Linear(d, 1) for _ in range(n_num)])
        self.cat_out = nn.ModuleList([nn.Linear(d, k) for k in num_classes])
        self.n_tokens, self.d = n_num + len(num_classes), d

    def encode(self, x_num, x_cat):
        tokens = self.tokenizer(x_num, x_cat)
        return self.enc_mu(tokens), self.enc_logvar(tokens)

    def decode(self, z_flat):
        h = self.dec(z_flat.view(-1, self.n_tokens, self.d))
        num = torch.cat([head(h[:, j]) for j, head in enumerate(self.num_out)], 1)
        logits = [head(h[:, self.n_num + j]) for j, head in enumerate(self.cat_out)]
        return num, logits

    def loss(self, x_num, x_cat, beta):
        mu, logvar = self.encode(x_num, x_cat)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        num, logits = self.decode(z.flatten(1))
        mse = ((num - x_num) ** 2).mean()
        ce = sum(F.cross_entropy(lg, x_cat[:, j]) for j, lg in enumerate(logits)) / len(logits)
        kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).mean()
        return mse + ce + beta * kl, mse + ce


class EDMDiffusion(nn.Module):
    """EDM (Karras et al.) denoiser on standardized latent vectors, conditioned on income."""

    def __init__(self, dim, config):
        super().__init__()
        self.config = config
        self.net = Denoiser(dim, dim, config["hidden"], config["dim_t"])

    def denoise(self, x, sigma, y):
        sd = self.config["sigma_data"]
        s = sigma[:, None]
        c_skip = sd ** 2 / (s ** 2 + sd ** 2)
        c_out = s * sd / (s ** 2 + sd ** 2).sqrt()
        c_in = 1 / (s ** 2 + sd ** 2).sqrt()
        # Noise level enters through the same sinusoidal embedding, scaled to a useful range.
        c_noise = sigma.log() / 4 * 1000
        return c_skip * x + c_out * self.net(c_in * x, c_noise, y)

    def per_row_loss(self, z, y, sigma=None, generator=None, return_denoised=False):
        """EDM's weighted denoising loss of each row; sigma=None draws the training noise levels."""
        c = self.config
        if sigma is None:
            sigma = (torch.randn(len(z), generator=generator) * c["p_std"] + c["p_mean"]).exp()
        noise = torch.randn(z.shape, generator=generator) * sigma[:, None]
        denoised = self.denoise(z + noise, sigma, y)
        weight = (sigma ** 2 + c["sigma_data"] ** 2) / (sigma * c["sigma_data"]) ** 2
        loss = weight * ((denoised - z) ** 2).mean(1)
        return (loss, denoised) if return_denoised else loss

    def sigmas(self, steps):
        c = self.config
        i = torch.arange(steps, dtype=torch.float32)
        inv = 1 / c["rho"]
        sig = (c["sigma_max"] ** inv + i / (steps - 1) * (c["sigma_min"] ** inv - c["sigma_max"] ** inv)) ** c["rho"]
        return torch.cat([sig, torch.zeros(1)])

    @torch.no_grad()
    def sample(self, y, dim, steps, generator=None, start=None):
        """Deterministic Heun solver of the probability-flow ODE. Randomness is only the start noise,
        so a sample can be inverted back to its noise (useful for the reverse-engineering study)."""
        sig = self.sigmas(steps)
        x = torch.randn((len(y), dim), generator=generator) * sig[0] if start is None else start
        for i in range(steps):
            s, s_next = sig[i], sig[i + 1]
            d = (x - self.denoise(x, s.expand(len(y)), y)) / s
            x_next = x + (s_next - s) * d
            if s_next > 0:
                d2 = (x_next - self.denoise(x_next, s_next.expand(len(y)), y)) / s_next
                x_next = x + (s_next - s) * 0.5 * (d + d2)
            x = x_next
        return x


# ------------------------------------------------------------ training ----

def _ema_update(ema, model, rate):
    with torch.no_grad():
        for e, p in zip(ema.parameters(), model.parameters()):
            e.mul_(rate).add_(p.detach(), alpha=1 - rate)


def _clone(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _snapshot_points(total):
    return sorted({max(1, round(total * q)) for q in (0.25, 0.5, 0.75, 1.0)})


class Sampler:
    """Draws rows from a trained diffusion step. Optional rule filter: broken rows are redrawn
    for the same label, up to 50 rounds, as in ladder.generate."""

    def __init__(self, saved, use="selected"):
        self.saved = saved
        self.config = saved["config"]
        self.encoder = saved["encoder"]
        enc = self.encoder
        if self.config["kind"] == "ddpm":
            self.model = GaussianMultinomialDiffusion(len(NUMERIC), enc.num_classes, self.config)
            self.model.load_state_dict(saved[f"{use}_state"])
        else:
            self.vae = VAE(len(NUMERIC), enc.num_classes, self.config)
            self.vae.load_state_dict(saved["vae_state"])
            self.dim = self.vae.n_tokens * self.vae.d
            self.model = EDMDiffusion(self.dim, self.config)
            self.model.load_state_dict(saved[f"{use}_state"])
            self.z_mean, self.z_std = saved["z_mean"], saved["z_std"]
        self.model.eval()

    @torch.no_grad()
    def _draw(self, labels, seed):
        g = torch.Generator().manual_seed(int(seed))
        y = torch.as_tensor(labels, dtype=torch.long)
        if self.config["kind"] == "ddpm":
            x_num, x_cat = self.model.sample(y, self.config["sample_steps"], g)
        else:
            self.vae.eval()
            z = self.model.sample(y, self.dim, self.config["sample_steps"], g)
            num, logits = self.vae.decode(z * self.z_std + self.z_mean)
            if self.config.get("decode", "argmax") == "sample":
                cat = [torch.multinomial(torch.softmax(lg, 1), 1, generator=g)[:, 0] for lg in logits]
            else:
                cat = [lg.argmax(1) for lg in logits]
            x_num, x_cat = num, torch.stack(cat, 1)
        return self.encoder.inverse(x_num.numpy(), x_cat.numpy())

    def __call__(self, labels, seed=0, stats=None, reject=None, draw=None):
        labels = np.asarray(labels, dtype=int)
        draw = draw or self._draw
        reject = self.config["reject"] if reject is None else reject
        frame = draw(labels, seed)
        if reject:
            bad, _ = rule_violations(frame)
            first_bad = int(bad.sum())
            for round_ in range(1, 51):
                if not bad.any():
                    break
                index = np.flatnonzero(bad)
                redraw = draw(labels[index], seed + 100003 * round_)
                frame.iloc[index] = redraw.to_numpy()
                bad[index] = rule_violations(redraw)[0]
            if stats is not None:
                stats["rejected_share"] = first_bad / max(len(frame), 1)
                stats["still_broken"] = int(bad.sum())
        frame["income"] = labels
        return frame


def load_diffusion(path, reject=None, use="selected"):
    """Rebuild a saved diffusion step. Returns a function: income labels -> synthetic frame."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    sampler = Sampler(saved, use=use)
    return lambda labels, seed=0, stats=None: sampler(labels, seed, stats, reject)


def _select_and_score(sampler_state, saved_base, select, schema, seed):
    saved = dict(saved_base, selected_state=sampler_state)
    sampler = Sampler(saved)
    synth = sampler(select["income"].to_numpy(), seed=42)
    return score(select.drop(columns=["income"]), synth.drop(columns=["income"]), schema)


def train_vae(vae, x_num, x_cat, config, seed, log):
    opt = torch.optim.Adam(vae.parameters(), lr=config["vae_lr"])
    g = torch.Generator().manual_seed(seed)
    beta, best_recon, patience, history = config["beta_max"], float("inf"), 0, []
    n = len(x_num)
    for epoch in range(1, config["vae_epochs"] + 1):
        perm = torch.randperm(n, generator=g)
        total = recon_total = 0.0
        for start in range(0, n, config["vae_batch"]):
            idx = perm[start:start + config["vae_batch"]]
            loss, recon = vae.loss(x_num[idx], x_cat[idx], beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
            recon_total += float(recon.detach()) * len(idx)
        recon_mean = recon_total / n
        history.append({"epoch": epoch, "loss": total / n, "recon": recon_mean, "beta": beta})
        # TabSyn's schedule: shrink the KL weight when reconstruction stops improving.
        if recon_mean < best_recon - 1e-4:
            best_recon, patience = recon_mean, 0
        else:
            patience += 1
            if patience >= config["beta_patience"] and beta > config["beta_min"]:
                beta, patience = max(beta * config["beta_decay"], config["beta_min"]), 0
        if epoch % 50 == 0 or epoch == config["vae_epochs"]:
            log(f"vae epoch {epoch}: loss {total / n:.4f} recon {recon_mean:.4f} beta {beta:.2e}")
    return history, opt.state_dict()


def train_step(step, config, seed, fit, select, test, schema, log, split=None, model_dir=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    encoder = TabEncoder(seed).fit(fit)
    x_num_np, x_cat_np = encoder.transform(fit)
    x_num, x_cat = torch.tensor(x_num_np), torch.tensor(x_cat_np)
    y = torch.tensor(fit["income"].to_numpy(), dtype=torch.long)
    saved = {"step": step, "seed": seed, "config": config, "encoder": encoder, "split": split,
             "torch_version": torch.__version__}
    started = time.time()

    if config["kind"] == "ddpm":
        model = GaussianMultinomialDiffusion(len(NUMERIC), encoder.num_classes, config)
        train_data = None
    else:
        vae = VAE(len(NUMERIC), encoder.num_classes, config)
        vae_history, vae_opt = train_vae(vae, x_num, x_cat, config, seed, log)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            mu, _ = vae.encode(x_num, x_cat)
        z = mu.flatten(1)
        z_mean, z_std = z.mean(0), z.std(0).clamp_min(1e-6)
        train_data = (z - z_mean) / z_std
        saved.update(vae_state=vae.state_dict(), vae_opt_state=vae_opt, vae_history=vae_history,
                     z_mean=z_mean, z_std=z_std)
        model = EDMDiffusion(train_data.shape[1], config)
        log(f"{step} seed {seed}: VAE done ({time.time() - started:.0f}s)")

    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1 - i / config["train_steps"])
    g = torch.Generator().manual_seed(seed)
    n, batch = len(y), config["batch_size"]
    losses, history, snapshots, best = [], [], {}, None
    keep = _snapshot_points(config["train_steps"])
    running = {"loss": 0.0, "rule": 0.0, "count": 0}
    for it in range(1, config["train_steps"] + 1):
        idx = torch.randint(n, (batch,), generator=g)
        if config["kind"] == "ddpm":
            t = torch.randint(1, model.T + 1, (batch,), generator=g)
            gauss, multi, probs = model.per_row_loss(x_num[idx], x_cat[idx], y[idx], t, g, return_probs=True)
            loss = (gauss + multi).mean()
            rule_probs = encoder.cat_probs(probs)
        else:
            per_row, denoised = model.per_row_loss(train_data[idx], y[idx], generator=g, return_denoised=True)
            loss = per_row.mean()
            rule_probs = None
            if config["rule_weight"]:
                _num, logits = vae.decode(denoised * z_std + z_mean)
                rule_probs = encoder.cat_probs([torch.softmax(lg, 1) for lg in logits])
        rule = expected_violations(rule_probs, encoder.levels).mean() if (config["rule_weight"] and rule_probs) else torch.tensor(0.0)
        total = loss + config["rule_weight"] * rule
        opt.zero_grad()
        total.backward()
        opt.step()
        sched.step()
        _ema_update(ema, model, config["ema"])
        running["loss"] += float(loss.detach())
        running["rule"] += float(rule.detach())
        running["count"] += 1
        if it % 500 == 0 or it == config["train_steps"]:
            losses.append({"step": it, "loss": running["loss"] / running["count"],
                           "rule": running["rule"] / running["count"]})
            running = {"loss": 0.0, "rule": 0.0, "count": 0}
        if it in keep:
            snapshots[it] = {"model_state": _clone(model), "ema_state": _clone(ema)}
        if it % config["score_every"] == 0 or it == config["train_steps"]:
            if running["count"]:
                losses.append({"step": it, "loss": running["loss"] / running["count"],
                               "rule": running["rule"] / running["count"]})
                running = {"loss": 0.0, "rule": 0.0, "count": 0}
            row = {"step": it, "seconds": round(time.time() - started, 1),
                   **_select_and_score(_clone(ema), saved, select, schema, seed)}
            history.append(row)
            log(f"{step} seed {seed} step {it}: loss {losses[-1]['loss']:.4f} auc {row['selection_auc']:.4f} "
                f"ks {row['ks']:.3f} tv {row['tv']:.3f} viol {row['violations']} ({row['seconds']}s)")
            key = (row["violations"] > 0, row["selection_auc"], it)
            if best is None or key < best["key"]:
                best = {"key": key, "row": row, "state": _clone(ema)}

    saved.update(
        selected_state=best["state"], selected_step=best["row"]["step"], ema_state=_clone(ema),
        final={"model_state": _clone(model), "ema_state": _clone(ema), "opt_state": opt.state_dict(),
               "sched_state": sched.state_dict(), "step": config["train_steps"],
               "generator_state": g.get_state()},
        snapshots=snapshots, losses=losses, history=history, train_seconds=round(time.time() - started, 1),
    )
    model_dir = Path(model_dir or MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(saved, model_dir / f"{step}_seed{seed}.pt")
    if test is None:
        # No test rows during training (ACS protocol): scoring happens once, later, in acs.py.
        return {"rung": step, "seed": seed,
                "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in config.items()},
                "selected": best["row"], "history": history, "train_seconds": saved["train_seconds"]}

    # Final scoring, once, after selection. Same draws and seeds as ladder.train_rung.
    sampler = Sampler(saved)
    select_x = select.drop(columns=["income"])
    dev = sampler(select["income"].to_numpy(), seed=7).drop(columns=["income"])
    t0 = time.time()
    test_synth = sampler(test["income"].to_numpy(), seed=9000 + seed).drop(columns=["income"])
    sample_seconds = time.time() - t0
    final = final_detection(select_x, dev, test.drop(columns=["income"]), test_synth, schema)
    fit_labels = fit["income"].to_numpy()
    utility_synth = sampler(fit_labels, seed=11).drop(columns=["income"])
    utility = utility_report(
        _design_matrix(fit.drop(columns=["income"]), schema), fit_labels,
        _design_matrix(utility_synth, schema), fit_labels,
        _design_matrix(test.drop(columns=["income"]), schema), test["income"].to_numpy(),
    )
    test_fidelity = score(test.drop(columns=["income"]), test_synth, schema)
    return {
        "rung": step, "seed": seed, "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in config.items()},
        "selected": best["row"], "history": history, "train_seconds": saved["train_seconds"],
        "final_detection_auc": final["mean_auc"], "utility_ratio": utility["ratio"],
        "utility_auc_real": utility["auc_real"], "utility_auc_synthetic": utility["auc_synthetic"],
        "test_ks": test_fidelity["ks"], "test_tv": test_fidelity["tv"], "test_coverage": test_fidelity["coverage"],
        "test_violations": test_fidelity["violations"],
        "sample_seconds_per_1k": 1000 * sample_seconds / len(test),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", nargs="+", default=list(STEPS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--train-steps", type=int, default=None, help="override for smoke tests")
    parser.add_argument("--vae-epochs", type=int, default=None, help="override for smoke tests")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = load_adult()
    schema = schema_from_frame(frame, NUMERIC, CATEGORICAL)
    torch.set_num_threads(int(__import__("os").environ.get("LADDER_THREADS", "1")))
    for step in args.steps:
        for seed in args.seeds:
            out = out_dir / f"{step}_seed{seed}.json"
            smoke = args.train_steps is not None or args.vae_epochs is not None
            if out.exists() and not smoke:
                print("skip", out.name, flush=True)
                continue
            config = dict(STEPS[step])
            if args.train_steps:
                config["train_steps"] = args.train_steps
                config["score_every"] = args.train_steps
            if args.vae_epochs and "vae_epochs" in config:
                config["vae_epochs"] = args.vae_epochs
            fit, select, test = split_frames(frame, seed)
            result = train_step(step, config, seed, fit, select, test, schema, lambda m: print(m, flush=True),
                                split=split_indices(frame, seed), model_dir=args.model_dir)
            out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
            print(f"DONE {step} seed {seed}: final AUC {result['final_detection_auc']:.4f} "
                  f"utility {result['utility_ratio']:.3f}", flush=True)


if __name__ == "__main__":
    # Run through the importable module so saved objects unpickle as diffusion.TabEncoder.
    import diffusion

    diffusion.main()

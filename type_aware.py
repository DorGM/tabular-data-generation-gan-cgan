"""Type-aware conditional generator.

This is a new model. It does not replace Generator or ConditionalGenerator.
Categorical attributes are one Gumbel-Softmax head each. Discrete education-num
is its own head. Age and hours are bounded continuous values rounded to integers
only at hard sampling. Capital gain and loss use a zero gate plus a positive
magnitude. The backbone is a plain MLP on noise plus a one-hot income label.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from main import ConditionalDiscriminator, build_optimizer, get_device, set_seed


def temperature_at(epoch, epochs, start=1.0, end=0.5):
    """Linear schedule. Epoch 0 uses `start`. The last epoch uses `end`."""
    if epochs <= 1:
        return float(end)
    progress = epoch / (epochs - 1)
    return float(start + (end - start) * progress)


def gumbel_softmax(logits, temperature, hard=False, straight_through=False):
    """Differentiable categorical sample.

    Training uses soft samples. Hard sampling is argmax and does not add noise,
    so evaluation is deterministic. Straight-through returns a one-hot value
    whose gradient is the soft sample's gradient. It is off unless requested.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if hard and not straight_through:
        index = torch.argmax(logits, dim=-1)
        return F.one_hot(index, num_classes=logits.shape[-1]).to(dtype=logits.dtype)
    uniform = torch.rand(logits.shape, device=logits.device, dtype=logits.dtype).clamp(1e-6, 1.0 - 1e-6)
    gumbel = -torch.log(-torch.log(uniform))
    soft = torch.softmax((logits + gumbel) / temperature, dim=-1)
    if not hard and not straight_through:
        return soft
    index = torch.argmax(soft, dim=-1)
    hard_sample = F.one_hot(index, num_classes=logits.shape[-1]).to(dtype=logits.dtype)
    return hard_sample.detach() - soft.detach() + soft


def fit_type_spec(original_frame, bundle):
    """Fit transforms on the fit slice only. Selection and test rows are not read."""
    fit = original_frame.iloc[np.asarray(bundle["train_index"])].reset_index(drop=True)
    numeric = {}
    for column in bundle["numeric_columns"]:
        values = fit[column].to_numpy(dtype=float)
        low = float(values.min())
        high = float(values.max())
        if not math.isfinite(low) or not math.isfinite(high) or high < low:
            raise ValueError(f"Bad fit range for {column}.")
        numeric[column] = {"min": low, "max": high}
    levels = sorted(set(np.rint(fit["education-num"].to_numpy(dtype=float)).astype(int).tolist()))
    if not levels:
        raise ValueError("education-num has no fit-slice levels.")
    return {
        "numeric": numeric,
        "education_levels": levels,
        "groups": bundle["categorical_groups"],
        "feature_names": list(bundle["feature_names"]),
        "numeric_columns": list(bundle["numeric_columns"]),
        "fit_rows": int(len(fit)),
    }


def _encode_unit(value, low, high):
    if high == low:
        return torch.zeros_like(value)
    return (value - low) / (high - low)


class TypeAwareConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, label_dim, spec, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.label_dim = int(label_dim)
        self.spec = spec
        self.output_dim = len(spec["feature_names"])
        self.backbone = nn.Sequential(
            nn.Linear(self.latent_dim + self.label_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.category_heads = nn.ModuleList(
            nn.Linear(hidden_dim, len(group["labels"])) for group in spec["groups"]
        )
        self.education_head = nn.Linear(hidden_dim, len(spec["education_levels"]))
        self.age_head = nn.Linear(hidden_dim, 1)
        self.hours_head = nn.Linear(hidden_dim, 1)
        self.fnlwgt_head = nn.Linear(hidden_dim, 1)
        self.gain_zero_head = nn.Linear(hidden_dim, 1)
        self.gain_magnitude_head = nn.Linear(hidden_dim, 1)
        self.loss_zero_head = nn.Linear(hidden_dim, 1)
        self.loss_magnitude_head = nn.Linear(hidden_dim, 1)
        self._name_index = {name: index for index, name in enumerate(spec["feature_names"])}

    def forward(self, noise, labels, temperature=1.0, hard=None, straight_through=False):
        if hard is None:
            hard = not self.training
        hidden = self.backbone(torch.cat([noise, labels], dim=1))
        parts = {}
        for head, group in zip(self.category_heads, self.spec["groups"]):
            sample = gumbel_softmax(head(hidden), temperature, hard=hard, straight_through=straight_through)
            for offset, index in enumerate(group["indexes"]):
                parts[index] = sample[:, offset]
        parts[self._name_index["education-num"]] = self._education(hidden, temperature, hard, straight_through)
        parts[self._name_index["age"]] = self._bounded_integer(self.age_head(hidden), "age", hard)
        parts[self._name_index["hours-per-week"]] = self._bounded_integer(
            self.hours_head(hidden), "hours-per-week", hard,
        )
        parts[self._name_index["fnlwgt"]] = self._log_span(self.fnlwgt_head(hidden), "fnlwgt")
        parts[self._name_index["capital-gain"]] = self._zero_inflated(
            self.gain_zero_head(hidden), self.gain_magnitude_head(hidden), "capital-gain", hard,
        )
        parts[self._name_index["capital-loss"]] = self._zero_inflated(
            self.loss_zero_head(hidden), self.loss_magnitude_head(hidden), "capital-loss", hard,
        )
        missing = [index for index in range(self.output_dim) if index not in parts]
        if missing:
            raise RuntimeError(f"Type-aware output is missing feature indexes {missing}.")
        return torch.stack([parts[index] for index in range(self.output_dim)], dim=1)

    def _education(self, hidden, temperature, hard, straight_through):
        levels = hidden.new_tensor(self.spec["education_levels"], dtype=hidden.dtype)
        sample = gumbel_softmax(
            self.education_head(hidden), temperature, hard=hard, straight_through=straight_through,
        )
        value = sample.matmul(levels)
        bounds = self.spec["numeric"]["education-num"]
        return _encode_unit(value, bounds["min"], bounds["max"]).view(-1)

    def _bounded_integer(self, logit, column, hard):
        bounds = self.spec["numeric"][column]
        unit = torch.sigmoid(logit.view(-1))
        value = bounds["min"] + unit * (bounds["max"] - bounds["min"])
        if hard:
            value = torch.round(value)
            value = torch.clamp(value, min=bounds["min"], max=bounds["max"])
        return _encode_unit(value, bounds["min"], bounds["max"])

    def _log_span(self, logit, column):
        bounds = self.spec["numeric"][column]
        low = math.log(max(bounds["min"], 1e-6))
        high = math.log(max(bounds["max"], low + 1e-6))
        log_value = low + torch.sigmoid(logit.view(-1)) * (high - low)
        return _encode_unit(torch.exp(log_value), bounds["min"], bounds["max"])

    def _zero_inflated(self, zero_logit, magnitude_logit, column, hard):
        bounds = self.spec["numeric"][column]
        probability_zero = torch.sigmoid(zero_logit.view(-1))
        magnitude = torch.sigmoid(magnitude_logit.view(-1)) * max(bounds["max"], 0.0)
        if hard:
            value = torch.where(probability_zero > 0.5, torch.zeros_like(magnitude), magnitude)
        else:
            value = (1.0 - probability_zero) * magnitude
        return _encode_unit(value, bounds["min"], bounds["max"])


def train_type_aware_cgan(
    x_train, y_train, spec, config, seed=0, device=None, temperature_fn=temperature_at,
    log_every=10, on_epoch=None, resume=None, resume_path=None,
):
    """BCE training with the original conditional discriminator. The generator is type-aware.

    This function does not score the test split and does not restore a selected epoch.
    Callers keep every checkpoint they write.
    """
    device = get_device() if device is None else torch.device(device)
    set_seed(seed)
    x_train = x_train.to(device)
    y_train = y_train.to(device).view(-1)
    label_dim = 2
    latent_dim = int(config["latent_dim"])
    batch_size = int(config["batch_size"])
    generator = TypeAwareConditionalGenerator(
        latent_dim, label_dim, spec, hidden_dim=int(config["hidden_dim"]), dropout=float(config["dropout"]),
    ).to(device)
    discriminator = ConditionalDiscriminator(
        x_train.shape[1], label_dim, hidden_dim=int(config["hidden_dim"]), dropout=float(config["dropout"]),
    ).to(device)
    criterion = nn.BCELoss()
    optimizer_g = build_optimizer(config["optimizer"], generator.parameters(), config["lr"], config["weight_decay"])
    optimizer_d = build_optimizer(config["optimizer"], discriminator.parameters(), config["lr"], config["weight_decay"])
    cpu_generator = torch.Generator()
    cpu_generator.manual_seed(seed)
    d_losses = []
    g_losses = []
    n_rows = x_train.shape[0]
    epochs = int(config["epochs"])
    start_epoch = 0
    if resume is not None:
        generator.load_state_dict(resume["generator"])
        discriminator.load_state_dict(resume["discriminator"])
        optimizer_g.load_state_dict(resume["optimizer_g"])
        optimizer_d.load_state_dict(resume["optimizer_d"])
        cpu_generator.set_state(resume["cpu_generator_state"])
        torch.set_rng_state(resume["torch_rng"])
        np.random.set_state(resume["numpy_rng"])
        d_losses = list(resume["d_losses"])
        g_losses = list(resume["g_losses"])
        start_epoch = int(resume["next_epoch"])
    for epoch in range(start_epoch, epochs):
        temperature = float(temperature_fn(epoch, epochs))
        permutation = torch.randperm(n_rows, generator=cpu_generator)
        d_sum = 0.0
        g_sum = 0.0
        batches = 0
        generator.train()
        discriminator.train()
        for start in range(0, n_rows, batch_size):
            columns = permutation[start:start + batch_size]
            real = x_train[columns]
            rows = real.shape[0]
            real_labels = _one_hot(y_train[columns], label_dim, device)
            fake_index = torch.randint(0, label_dim, (rows,), generator=cpu_generator)
            fake_labels = _one_hot(fake_index.to(device), label_dim, device)
            real_target = torch.ones(rows, 1, device=device)
            fake_target = torch.zeros(rows, 1, device=device)
            noise = torch.randn(rows, latent_dim, generator=cpu_generator).to(device)
            fake = generator(noise, fake_labels, temperature=temperature, hard=False)
            loss_d = criterion(discriminator(real, real_labels), real_target) + criterion(
                discriminator(fake.detach(), fake_labels), fake_target,
            )
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()
            loss_g = criterion(discriminator(fake, fake_labels), real_target)
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()
            d_sum += float(loss_d.item())
            g_sum += float(loss_g.item())
            batches += 1
        d_losses.append(d_sum / batches)
        g_losses.append(g_sum / batches)
        if not math.isfinite(d_losses[-1]) or not math.isfinite(g_losses[-1]):
            break
        if on_epoch is not None and ((epoch + 1) % 10 == 0 or epoch + 1 == epochs):
            generator.eval()
            on_epoch(epoch, generator, temperature, d_losses[-1], g_losses[-1])
            generator.train()
        if resume_path is not None and ((epoch + 1) % 10 == 0 or epoch + 1 == epochs):
            _write_resume(
                resume_path, generator, discriminator, optimizer_g, optimizer_d,
                cpu_generator, d_losses, g_losses, epoch + 1,
            )
        if log_every and (epoch + 1) % int(log_every) == 0:
            print(
                f"type-aware cgan epoch {epoch + 1} temperature {temperature:.3f} "
                f"d_loss {d_losses[-1]:.4f} g_loss {g_losses[-1]:.4f}",
                flush=True,
            )
    generator.eval()
    return {
        "generator": generator,
        "discriminator": discriminator,
        "d_loss": d_losses,
        "g_loss": g_losses,
        "config": dict(config),
        "seed": seed,
        "kind": "type_aware_cgan",
        "label_dim": label_dim,
        "temperature_schedule": {"start": 1.0, "end": 0.5, "interpolation": "linear"},
    }


def _write_resume(path, generator, discriminator, optimizer_g, optimizer_d, cpu_generator, d_losses, g_losses, next_epoch):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "generator": {key: value.detach().cpu() for key, value in generator.state_dict().items()},
            "discriminator": {key: value.detach().cpu() for key, value in discriminator.state_dict().items()},
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "cpu_generator_state": cpu_generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "d_losses": list(d_losses),
            "g_losses": list(g_losses),
            "next_epoch": int(next_epoch),
        },
        temporary,
    )
    temporary.replace(path)


def _one_hot(labels, label_dim, device):
    index = labels.long().view(-1, 1)
    encoded = torch.zeros(index.shape[0], label_dim, device=device)
    encoded.scatter_(1, index, 1.0)
    return encoded

"""GAN and conditional GAN for the supplied Adult ARFF table.

Two generator heads are available. 'linear' has no final activation and is the
variant from the available notebook; numeric columns are standardized. 'sigmoid'
is a reconstruction of the missing notebook: the same hidden stack plus a final
sigmoid, with numeric columns scaled to [0, 1]. The unconditional generator does
not emit an income label, so efficacy is defined only for the conditional model.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PATH = PROJECT_DIR / "data" / "adult.arff"
SPLIT_SEEDS = (42, 68, 124)
NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
DROP_COLUMNS = ["education"]
TARGET_COLUMN = "income"
GAN_EPOCHS = (10, 20, 30)
CGAN_EPOCHS = (15, 25, 40)
INCOME_MAP = {"<=50K": 0, ">50K": 1}
SCREEN_EPOCHS = 3


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "?"
    return str(value)


def load_adult(path=None):
    """Parse the supplied Adult ARFF. Nominal fields are strings, including '?'."""
    path = DEFAULT_PATH if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Adult ARFF not found at {path}. Place the supplied file there "
            "(32,561 rows, 15 attributes, including income). Do not substitute a "
            "different Adult CSV."
        )
    raw, _meta = arff.loadarff(path)
    frame = pd.DataFrame(raw)
    expected = NUMERIC_COLUMNS + DROP_COLUMNS + [
        "workclass",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        TARGET_COLUMN,
    ]
    missing = [column for column in expected if column not in frame.columns]
    if missing or frame.shape[1] != 15:
        raise ValueError(
            f"{path} does not look like the supplied Adult ARFF "
            f"(columns={list(frame.columns)})."
        )
    for column in frame.columns:
        if column in NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        else:
            frame[column] = frame[column].map(_decode)
    return frame


def prepare_experiment(frame, seed, output_activation, test_size=0.2, selection_fraction=0.2):
    """Stratified 80/20 train/test split, then a selection slice inside train.

    The scaler and dummy columns are fit on the fit slice only. `x_select` is
    for detection-based model selection. `x_test` is not used for that choice.
    output_activation 'sigmoid' uses MinMaxScaler. 'linear' uses StandardScaler.
    """
    if output_activation not in {"sigmoid", "linear"}:
        raise ValueError("output_activation must be 'sigmoid' or 'linear'.")
    data = frame.copy()
    decoded_income = data[TARGET_COLUMN].map(_decode)
    mapped_income = decoded_income.map(INCOME_MAP)
    if mapped_income.isna().any():
        unknown = sorted(set(decoded_income) - set(INCOME_MAP))
        raise ValueError(f"Unmapped income values: {unknown}")
    data[TARGET_COLUMN] = mapped_income.astype(int)
    data = data.drop(columns=DROP_COLUMNS)
    y = data[TARGET_COLUMN]
    features = data.drop(columns=[TARGET_COLUMN])
    missing_numeric = [column for column in NUMERIC_COLUMNS if column not in features.columns]
    if missing_numeric:
        raise ValueError(f"Missing numeric columns: {missing_numeric}")
    categorical = [column for column in features.columns if column not in NUMERIC_COLUMNS]
    for column in categorical:
        features[column] = features[column].map(_decode)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_index, test_index = next(splitter.split(features, y))
    y_development = y.iloc[train_index]
    fit_pos, select_pos = _fit_select_positions(y_development, selection_fraction, seed)
    fit_index = np.asarray(train_index)[fit_pos]
    select_index = np.asarray(train_index)[select_pos]

    def take(index):
        return features.iloc[index].reset_index(drop=True), y.iloc[index].reset_index(drop=True)

    x_fit, y_fit = take(fit_index)
    x_select, y_select = take(select_index)
    x_test, y_test = take(test_index)

    fit_encoded = pd.get_dummies(x_fit, columns=categorical, dtype=float)
    select_encoded = pd.get_dummies(x_select, columns=categorical, dtype=float)
    test_encoded = pd.get_dummies(x_test, columns=categorical, dtype=float)
    select_encoded = select_encoded.reindex(columns=fit_encoded.columns, fill_value=0.0)
    test_encoded = test_encoded.reindex(columns=fit_encoded.columns, fill_value=0.0)
    dummy_columns = [column for column in fit_encoded.columns if column not in NUMERIC_COLUMNS]
    ordered = list(NUMERIC_COLUMNS) + dummy_columns
    fit_encoded = fit_encoded.loc[:, ordered].copy()
    select_encoded = select_encoded.loc[:, ordered].copy()
    test_encoded = test_encoded.loc[:, ordered].copy()

    if output_activation == "sigmoid":
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        scaling = "minmax"
    else:
        scaler = StandardScaler()
        scaling = "standard"
    fit_encoded.loc[:, NUMERIC_COLUMNS] = scaler.fit_transform(fit_encoded.loc[:, NUMERIC_COLUMNS])
    select_encoded.loc[:, NUMERIC_COLUMNS] = scaler.transform(select_encoded.loc[:, NUMERIC_COLUMNS])
    test_encoded.loc[:, NUMERIC_COLUMNS] = scaler.transform(test_encoded.loc[:, NUMERIC_COLUMNS])

    def tensor_frame(encoded):
        return torch.tensor(encoded.to_numpy(), dtype=torch.float32)

    groups = _categorical_groups(ordered, categorical)
    group_width = sum(len(group["indexes"]) for group in groups)
    if group_width != len(ordered) - len(NUMERIC_COLUMNS):
        raise RuntimeError("Categorical groups do not cover the dummy columns.")

    return {
        "seed": int(seed),
        "output_activation": output_activation,
        "scaling": scaling,
        "feature_names": ordered,
        "numeric_columns": list(NUMERIC_COLUMNS),
        "categorical_columns": categorical,
        "categorical_groups": groups,
        "x_train": tensor_frame(fit_encoded),
        "y_train": torch.tensor(y_fit.to_numpy(), dtype=torch.float32),
        "x_select": tensor_frame(select_encoded),
        "y_select": torch.tensor(y_select.to_numpy(), dtype=torch.float32),
        "x_test": tensor_frame(test_encoded),
        "y_test": torch.tensor(y_test.to_numpy(), dtype=torch.float32),
        "scaler": scaler,
        "train_index": np.asarray(fit_index),
        "select_index": np.asarray(select_index),
        "test_index": np.asarray(test_index),
    }


def _fit_select_positions(y_development, selection_fraction, seed):
    """Positions inside the training split. The test rows are not in this vector."""
    if not 0 < selection_fraction < 1:
        raise ValueError("selection_fraction must be between 0 and 1.")
    y_values = np.asarray(y_development)
    n_rows = len(y_values)
    n_select = int(round(n_rows * selection_fraction))
    n_select = min(max(n_select, 1), n_rows - 1)
    n_classes = len(set(y_values.tolist()))
    n_fit = n_rows - n_select
    if n_select >= n_classes and n_fit >= n_classes:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=n_select, random_state=seed)
        fit_pos, select_pos = next(splitter.split(np.zeros((n_rows, 1)), y_values))
        return fit_pos, select_pos
    generator = np.random.default_rng(seed)
    order = generator.permutation(n_rows)
    return order[n_select:], order[:n_select]


def _categorical_groups(feature_names, categorical_columns):
    """Dummy-column groups in feature order. Longest attribute name wins a prefix tie."""
    names = list(feature_names)
    groups = []
    claimed = set()
    for column in sorted(categorical_columns, key=len, reverse=True):
        prefix = column + "_"
        indexes = []
        labels = []
        for index, name in enumerate(names):
            if index in claimed or not name.startswith(prefix):
                continue
            indexes.append(index)
            labels.append(name[len(prefix):])
        if not indexes:
            raise ValueError(f"No dummy columns for categorical field '{column}'.")
        claimed.update(indexes)
        groups.append({"name": column, "indexes": indexes, "labels": labels})
    groups.sort(key=lambda group: group["indexes"][0])
    return groups


# Absolute bounds after inverse scaling. A generator that leaves this box is not a usable table.
PLAUSIBLE_NUMERIC = {
    "age": (0.0, 120.0),
    "fnlwgt": (0.0, 2_000_000.0),
    "education-num": (0.0, 20.0),
    "capital-gain": (0.0, 200_000.0),
    "capital-loss": (0.0, 10_000.0),
    "hours-per-week": (0.0, 168.0),
}


def inverse_numeric(values, feature_names, scaler):
    """Map the numeric columns back with the train-fitted scaler. Dummies are dropped."""
    array = np.asarray(values, dtype=float)
    if torch.is_tensor(values):
        array = values.detach().cpu().numpy()
    frame = pd.DataFrame(array, columns=list(feature_names))
    restored = scaler.inverse_transform(frame.loc[:, NUMERIC_COLUMNS])
    return restored


def decode_rows(values, feature_names, numeric_columns, categorical_groups, scaler):
    """Inverse-transform numerics and hard-decode each categorical group.

    Hard decode is argmax inside the group. An all-zero group stays all-zero,
    which is how an unseen real category appears after reindexing. Linear and
    sigmoid generators then share original numeric units plus a 0/1 category code.
    """
    if torch.is_tensor(values):
        array = values.detach().cpu().numpy().astype(float)
    else:
        array = np.asarray(values, dtype=float)
    names = list(feature_names)
    numeric_index = [names.index(column) for column in numeric_columns]
    numeric = np.asarray(scaler.inverse_transform(array[:, numeric_index]), dtype=float)
    blocks = []
    unit_flags = []
    extreme_flags = []
    for group in categorical_groups:
        block = array[:, group["indexes"]]
        decoded = np.zeros_like(block)
        present = np.any(block != 0.0, axis=1)
        choice = np.argmax(block, axis=1)
        rows = np.arange(len(block))[present]
        decoded[rows, choice[present]] = 1.0
        blocks.append(decoded)
        unit_flags.append((block < 0.0) | (block > 1.0))
        extreme_flags.append((block < -0.5) | (block > 1.5))
    hard = np.concatenate(blocks, axis=1) if blocks else np.zeros((len(array), 0))
    unit = np.concatenate(unit_flags, axis=1) if unit_flags else np.zeros((len(array), 0), dtype=bool)
    extreme = np.concatenate(extreme_flags, axis=1) if extreme_flags else np.zeros((len(array), 0), dtype=bool)
    return {
        "numeric": numeric,
        "hard": hard,
        "common": np.concatenate([numeric, hard], axis=1),
        "finite": bool(np.isfinite(array).all() and np.isfinite(numeric).all()),
        "categorical_out_of_unit_fraction": float(unit.mean()) if unit.size else 0.0,
        "categorical_extreme_fraction": float(extreme.mean()) if extreme.size else 0.0,
    }


def categorical_frequency_table(hard, categorical_groups):
    """Mean of hard 0/1 codes, one table per original categorical attribute."""
    tables = {}
    start = 0
    for group in categorical_groups:
        width = len(group["indexes"])
        block = np.asarray(hard)[:, start:start + width]
        start += width
        means = block.mean(axis=0)
        tables[group["name"]] = {
            label: float(means[index]) for index, label in enumerate(group["labels"])
        }
    if start != np.asarray(hard).shape[1]:
        raise RuntimeError("Hard categorical width does not match the group list.")
    return tables


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=128, dropout=0.0, output_activation="linear"):
        super().__init__()
        if output_activation not in {"linear", "sigmoid"}:
            raise ValueError("output_activation must be 'linear' or 'sigmoid'.")
        self.output_activation = output_activation
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, noise):
        output = self.net(noise)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(output)
        return output


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        return self.net(features)


class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, label_dim, output_dim, hidden_dim=128, dropout=0.0, output_activation="linear"):
        super().__init__()
        if output_activation not in {"linear", "sigmoid"}:
            raise ValueError("output_activation must be 'linear' or 'sigmoid'.")
        self.output_activation = output_activation
        self.net = nn.Sequential(
            nn.Linear(latent_dim + label_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, noise, labels):
        output = self.net(torch.cat([noise, labels], dim=1))
        if self.output_activation == "sigmoid":
            return torch.sigmoid(output)
        return output


class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_dim, label_dim, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + label_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features, labels):
        return self.net(torch.cat([features, labels], dim=1))


def build_optimizer(name, parameters, lr, weight_decay):
    key = name.lower()
    if key == "adam":
        return torch.optim.Adam(parameters, lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)
    if key == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)
    if key == "sgd":
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{name}'. Use Adam, AdamW, or SGD.")


def _one_hot(labels, label_dim, device):
    index = labels.long().view(-1, 1)
    if int(index.max()) >= label_dim or int(index.min()) < 0:
        raise ValueError(f"Labels must be in 0..{label_dim - 1}.")
    encoded = torch.zeros(index.shape[0], label_dim, device=device)
    encoded.scatter_(1, index, 1.0)
    return encoded


def _noise(rows, latent_dim, generator, device):
    return torch.randn(rows, latent_dim, generator=generator).to(device)


def train_gan(x_train, config, seed=0, device=None, selection=None, log_every=10):
    """Alternating discriminator and generator steps. No income label is used or emitted.

    When `selection` is set, candidate checkpoints are scored on that validation
    slice only. Generator and discriminator losses are logged and are not the
    selection score. The returned generator is the best valid checkpoint.
    """
    device = get_device() if device is None else torch.device(device)
    set_seed(seed)
    x_train = x_train.to(device)
    latent_dim = int(config["latent_dim"])
    batch_size = int(config["batch_size"])
    generator = Generator(
        latent_dim,
        x_train.shape[1],
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
        output_activation=config["output_activation"],
    ).to(device)
    discriminator = Discriminator(
        x_train.shape[1],
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    criterion = nn.BCELoss()
    optimizer_g = build_optimizer(config["optimizer"], generator.parameters(), config["lr"], config["weight_decay"])
    optimizer_d = build_optimizer(config["optimizer"], discriminator.parameters(), config["lr"], config["weight_decay"])
    cpu_generator = torch.Generator()
    cpu_generator.manual_seed(seed)
    d_losses = []
    g_losses = []
    n_rows = x_train.shape[0]
    selection_state = _empty_selection_state()

    for epoch in range(int(config["epochs"])):
        permutation = torch.randperm(n_rows, generator=cpu_generator)
        d_sum = 0.0
        g_sum = 0.0
        batches = 0
        for start in range(0, n_rows, batch_size):
            columns = permutation[start:start + batch_size]
            real = x_train[columns]
            rows = real.shape[0]
            real_target = torch.ones(rows, 1, device=device)
            fake_target = torch.zeros(rows, 1, device=device)

            noise = _noise(rows, latent_dim, cpu_generator, device)
            fake = generator(noise)
            loss_d = criterion(discriminator(real), real_target) + criterion(discriminator(fake.detach()), fake_target)
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            noise = _noise(rows, latent_dim, cpu_generator, device)
            fake = generator(noise)
            loss_g = criterion(discriminator(fake), real_target)
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
        _consider_checkpoint(
            generator, epoch, "gan", None, config, selection, selection_state, device,
        )
        if log_every and (epoch + 1) % int(log_every) == 0:
            print(
                f"gan epoch {epoch + 1} d_loss {d_losses[-1]:.4f} g_loss {g_losses[-1]:.4f}",
                flush=True,
            )

    _restore_selection(generator, selection_state)
    result = {
        "generator": generator,
        "discriminator": discriminator,
        "d_loss": d_losses,
        "g_loss": g_losses,
        "loss_mean": (sum(d_losses) + sum(g_losses)) / (2.0 * len(d_losses)),
        "config": dict(config),
        "seed": seed,
        "kind": "gan",
    }
    _attach_selection(result, selection_state, selection)
    return result


def train_cgan(x_train, y_train, config, seed=0, device=None, selection=None, log_every=10):
    """Conditional steps. Labels are the income values of this split, one-hot encoded.

    Checkpoint selection uses `selection` only. Losses stay diagnostic. A conditional
    checkpoint is conditioned on the selection labels, not on invented targets.
    """
    device = get_device() if device is None else torch.device(device)
    set_seed(seed)
    x_train = x_train.to(device)
    y_train = y_train.to(device).view(-1)
    label_dim = int(y_train.max().item()) + 1
    if label_dim != 2:
        raise ValueError(f"Expected binary income labels, found label_dim={label_dim}.")
    latent_dim = int(config["latent_dim"])
    batch_size = int(config["batch_size"])
    generator = ConditionalGenerator(
        latent_dim,
        label_dim,
        x_train.shape[1],
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
        output_activation=config["output_activation"],
    ).to(device)
    discriminator = ConditionalDiscriminator(
        x_train.shape[1],
        label_dim,
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    criterion = nn.BCELoss()
    optimizer_g = build_optimizer(config["optimizer"], generator.parameters(), config["lr"], config["weight_decay"])
    optimizer_d = build_optimizer(config["optimizer"], discriminator.parameters(), config["lr"], config["weight_decay"])
    cpu_generator = torch.Generator()
    cpu_generator.manual_seed(seed)
    d_losses = []
    g_losses = []
    n_rows = x_train.shape[0]
    selection_state = _empty_selection_state()

    for epoch in range(int(config["epochs"])):
        permutation = torch.randperm(n_rows, generator=cpu_generator)
        d_sum = 0.0
        g_sum = 0.0
        batches = 0
        for start in range(0, n_rows, batch_size):
            columns = permutation[start:start + batch_size]
            real = x_train[columns]
            real_labels = _one_hot(y_train[columns], label_dim, device)
            rows = real.shape[0]
            real_target = torch.ones(rows, 1, device=device)
            fake_target = torch.zeros(rows, 1, device=device)
            fake_index = torch.randint(0, label_dim, (rows,), generator=cpu_generator)
            fake_labels = _one_hot(fake_index.to(device), label_dim, device)

            noise = _noise(rows, latent_dim, cpu_generator, device)
            fake = generator(noise, fake_labels)
            loss_real = criterion(discriminator(real, real_labels), real_target)
            loss_fake = criterion(discriminator(fake.detach(), fake_labels), fake_target)
            loss_d = loss_real + loss_fake
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
        _consider_checkpoint(
            generator, epoch, "cgan", label_dim, config, selection, selection_state, device,
        )
        if log_every and (epoch + 1) % int(log_every) == 0:
            print(
                f"cgan epoch {epoch + 1} d_loss {d_losses[-1]:.4f} g_loss {g_losses[-1]:.4f}",
                flush=True,
            )

    _restore_selection(generator, selection_state)
    result = {
        "generator": generator,
        "discriminator": discriminator,
        "d_loss": d_losses,
        "g_loss": g_losses,
        "loss_mean": (sum(d_losses) + sum(g_losses)) / (2.0 * len(d_losses)),
        "config": dict(config),
        "seed": seed,
        "kind": "cgan",
        "label_dim": label_dim,
    }
    _attach_selection(result, selection_state, selection)
    return result


def _as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _empty_selection_state():
    return {
        "history": [],
        "best_gap": float("inf"),
        "best_epoch": None,
        "best_state": None,
        "best_report": None,
    }


def _restore_selection(generator, state):
    if state["best_state"] is not None:
        generator.load_state_dict(state["best_state"])
    generator.eval()


def _attach_selection(result, state, selection):
    result["checkpoints"] = list(state["history"])
    result["selected_epoch"] = state["best_epoch"]
    result["selection"] = state["best_report"]
    if selection is None:
        result["acceptable"] = None
    else:
        result["acceptable"] = state["best_state"] is not None


def _sample_for_selection(generator, kind, label_dim, config, selection, device):
    """One synthetic row per selection row. cGAN uses that row's own income label."""
    latent_dim = int(config["latent_dim"])
    seed = int(selection.get("sample_seed", 42))
    real_x = selection["x_select"]
    n_rows = int(real_x.shape[0])
    if kind == "gan":
        return sample_gan(generator, n_rows, latent_dim, seed, device=device), None
    if kind == "cgan":
        labels = selection.get("y_select")
        if labels is None or len(labels) != n_rows:
            raise ValueError("cGAN selection needs the real label of every selection row.")
        return sample_cgan(generator, labels, latent_dim, label_dim, seed, device=device)
    raise ValueError(f"Unknown generator kind '{kind}'.")


def _consider_checkpoint(generator, epoch, kind, label_dim, config, selection, state, device):
    """Score a finished epoch on the selection slice. Losses are not consulted."""
    if selection is None:
        return
    epoch_number = epoch + 1
    every = int(selection.get("every", 10))
    if epoch_number % every != 0 and epoch_number != int(config["epochs"]):
        return
    was_training = generator.training
    generator.eval()
    try:
        synthetic, _labels = _sample_for_selection(
            generator, kind, label_dim, config, selection, device,
        )
        report = compare_real_synthetic(
            selection["x_select"],
            synthetic,
            selection["bundle"],
            n_splits=int(selection.get("n_splits", 4)),
            n_estimators=int(selection.get("n_estimators", 80)),
            random_state=int(selection.get("detector_seed", 42)),
            max_depth=selection.get("max_depth", 16),
        )
    finally:
        if was_training:
            generator.train()
    fidelity = report["fidelity"]
    row = {
        "epoch": epoch,
        "epoch_number": epoch_number,
        "mean_auc": report["mean_auc"],
        "fold_auc": report["fold_auc"],
        "gap": report["gap"],
        "valid": fidelity["valid"],
        "diversity": fidelity["diversity"],
        "numeric_mean_shift": fidelity["numeric_mean_shift"],
        "correlation_mae": fidelity["correlation_mae"],
        "categorical_mae": fidelity["categorical_mae"],
        "numeric_range_violation_fraction": fidelity["numeric_range_violation_fraction"],
        "plausible_violation_fraction": fidelity["plausible_violation_fraction"],
        "categorical_extreme_fraction": fidelity["categorical_extreme_fraction"],
        "categorical_out_of_unit_fraction": fidelity["categorical_out_of_unit_fraction"],
    }
    state["history"].append(row)
    checkpoint_dir = selection.get("checkpoint_dir")
    if checkpoint_dir:
        folder = Path(checkpoint_dir)
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(
            {key: value.detach().cpu().clone() for key, value in generator.state_dict().items()},
            folder / f"epoch_{epoch_number:04d}.pt",
        )
    if row["valid"] and row["gap"] < state["best_gap"]:
        state["best_gap"] = row["gap"]
        state["best_epoch"] = epoch
        state["best_state"] = {
            key: value.detach().clone() for key, value in generator.state_dict().items()
        }
        state["best_report"] = dict(row)
    print(
        f"  checkpoint epoch {epoch_number} gap {row['gap']:.4f} "
        f"auc {row['mean_auc']:.4f} valid {row['valid']}",
        flush=True,
    )


def statistical_fidelity(real_decoded, synth_decoded, numeric_columns):
    """Validity and distance checks on rows already mapped to the common decoded domain.

    A detection score near 0.5 does not override these gates. Collapsed rows, non-finite
    values, numeric values outside a loose neighborhood of the real selection slice, and
    categorical codes far outside the unit interval are rejected.
    """
    real_num = np.asarray(real_decoded["numeric"], dtype=float)
    synth_num = np.asarray(synth_decoded["numeric"], dtype=float)
    real_hard = np.asarray(real_decoded["hard"], dtype=float)
    synth_hard = np.asarray(synth_decoded["hard"], dtype=float)
    finite = bool(real_decoded["finite"] and synth_decoded["finite"])
    signature = np.concatenate([np.round(synth_num, 1), np.round(synth_hard, 0)], axis=1)
    diversity = float(np.unique(signature, axis=0).shape[0] / max(len(signature), 1))

    shifts = []
    std_ratios = []
    range_hits = 0
    plausible_hits = 0
    numeric_count = 0
    for index, column in enumerate(numeric_columns):
        real_col = real_num[:, index]
        synth_col = synth_num[:, index]
        scale = float(real_col.std()) + 1e-6
        shifts.append(abs(float(synth_col.mean() - real_col.mean())) / scale)
        std_ratios.append(float(synth_col.std()) / scale)
        span = max(float(real_col.max() - real_col.min()), 1.0)
        low = float(real_col.min()) - 0.5 * span
        high = float(real_col.max()) + 0.5 * span
        range_hits += int(np.sum((synth_col < low) | (synth_col > high)))
        bounds = PLAUSIBLE_NUMERIC[column]
        plausible_hits += int(np.sum((synth_col < bounds[0]) | (synth_col > bounds[1])))
        numeric_count += len(synth_col)
    if real_num.shape[1] >= 2 and len(real_num) >= 2 and len(synth_num) >= 2:
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_real = np.corrcoef(real_num, rowvar=False)
            corr_synth = np.corrcoef(synth_num, rowvar=False)
        correlation_mae = float(np.nanmean(np.abs(corr_real - corr_synth)))
    else:
        correlation_mae = 0.0
    if real_hard.size:
        categorical_mae = float(np.mean(np.abs(real_hard.mean(axis=0) - synth_hard.mean(axis=0))))
    else:
        categorical_mae = 0.0
    range_fraction = range_hits / max(numeric_count, 1)
    plausible_fraction = plausible_hits / max(numeric_count, 1)
    extreme_fraction = float(synth_decoded["categorical_extreme_fraction"])
    collapsed = diversity < 0.10 or (std_ratios and float(np.mean(std_ratios)) < 0.05)
    valid = bool(
        finite
        and not collapsed
        and range_fraction <= 0.02
        and plausible_fraction <= 0.02
        and extreme_fraction <= 0.25
    )
    return {
        "valid": valid,
        "diversity": diversity,
        "numeric_mean_shift": float(np.mean(shifts)) if shifts else 0.0,
        "correlation_mae": correlation_mae,
        "categorical_mae": categorical_mae,
        "numeric_range_violation_fraction": float(range_fraction),
        "plausible_violation_fraction": float(plausible_fraction),
        "categorical_extreme_fraction": extreme_fraction,
        "categorical_out_of_unit_fraction": float(synth_decoded["categorical_out_of_unit_fraction"]),
        "collapsed": bool(collapsed),
        "finite": finite,
    }


def compare_real_synthetic(
    real_x, synth_x, bundle, n_splits=4, n_estimators=80, random_state=42, max_depth=16,
):
    """Detection and fidelity on the same decoded rows, in one feature order."""
    if len(real_x) != len(synth_x):
        raise ValueError(
            "Real and synthetic batches must contain the same number of rows. "
            "Choose the real rows once and condition a cGAN on those labels."
        )
    real_decoded = decode_rows(
        real_x, bundle["feature_names"], bundle["numeric_columns"],
        bundle["categorical_groups"], bundle["scaler"],
    )
    synth_decoded = decode_rows(
        synth_x, bundle["feature_names"], bundle["numeric_columns"],
        bundle["categorical_groups"], bundle["scaler"],
    )
    if real_decoded["common"].shape != synth_decoded["common"].shape:
        raise RuntimeError("Decoded real and synthetic matrices do not share a shape.")
    fidelity = statistical_fidelity(real_decoded, synth_decoded, bundle["numeric_columns"])
    detection = detection_auc(
        real_decoded["common"], synth_decoded["common"],
        n_splits=n_splits, random_state=random_state,
        n_estimators=n_estimators, max_depth=max_depth,
    )
    return {
        "fidelity": fidelity,
        "mean_auc": detection["mean_auc"],
        "fold_auc": detection["fold_auc"],
        "gap": abs(detection["mean_auc"] - 0.5),
        "real_decoded": real_decoded,
        "synth_decoded": synth_decoded,
    }


def detection_gap(real_common, synth_common, n_splits=4, random_state=42, n_estimators=100, max_depth=None):
    """abs(ROC-AUC - 0.5). The two matrices must already be aligned in the common domain."""
    if len(real_common) != len(synth_common):
        raise ValueError(
            "Real and synthetic matrices must be the same length. "
            "Do not draw a separate subset for detection and for fidelity."
        )
    result = detection_auc(
        real_common, synth_common, n_splits=n_splits, random_state=random_state,
        n_estimators=n_estimators, max_depth=max_depth,
    )
    result["gap"] = abs(result["mean_auc"] - 0.5)
    return result


def select_best(runs, bundle, seed=42, n_estimators=80, max_depth=16, device=None):
    """Lowest selection-slice detection gap among valid generators.

    `bundle` is a prepare_experiment result. Scoring uses `x_select` and, for a
    cGAN, `y_select`. The test split is not read. Generator loss is not read.
    An unconditional generator is not given income labels. If every candidate
    fails the validity gates, nothing is selected.
    """
    if not runs:
        raise ValueError("No runs to select from.")
    real_x = bundle["x_select"]
    reports = []
    best = None
    best_gap = float("inf")
    for run in runs:
        if "x_test" in run or "y_test" in run:
            raise ValueError("A selection run must not carry the held-out test split.")
        selection = {
            "x_select": real_x,
            "y_select": bundle["y_select"] if run["kind"] == "cgan" else None,
            "bundle": bundle,
            "sample_seed": seed,
        }
        synthetic, labels = _sample_for_selection(
            run["generator"], run["kind"], run.get("label_dim"), run["config"], selection, device,
        )
        compared = compare_real_synthetic(
            real_x, synthetic, bundle, n_estimators=n_estimators,
            random_state=seed, max_depth=max_depth,
        )
        report = {
            "gap": compared["gap"],
            "mean_auc": compared["mean_auc"],
            "fold_auc": compared["fold_auc"],
            "fidelity": compared["fidelity"],
            "run": run,
            "conditioned_on_real_labels": labels is not None,
        }
        reports.append(report)
        if compared["fidelity"]["valid"] and compared["gap"] < best_gap:
            best_gap = compared["gap"]
            best = report
    if best is None:
        return {
            "acceptable": False,
            "reports": reports,
            "reason": (
                "Every candidate failed the validity gates, so none was selected. "
                "Detection gap is not a substitute for a usable table."
            ),
        }
    best = dict(best)
    best["acceptable"] = True
    best["reports"] = reports
    return best


@torch.no_grad()
def sample_gan(generator, n_rows, latent_dim, seed, device=None):
    """Feature rows only. Income is not attached."""
    device = get_device() if device is None else torch.device(device)
    generator.eval()
    cpu_generator = torch.Generator()
    cpu_generator.manual_seed(seed)
    noise = _noise(n_rows, latent_dim, cpu_generator, device)
    return generator(noise).cpu()


@torch.no_grad()
def sample_cgan(generator, labels, latent_dim, label_dim, seed, device=None):
    """Condition on the labels passed in. Those labels are returned with the rows."""
    device = get_device() if device is None else torch.device(device)
    generator.eval()
    labels = torch.as_tensor(labels).long().view(-1)
    cpu_generator = torch.Generator()
    cpu_generator.manual_seed(seed)
    noise = _noise(labels.shape[0], latent_dim, cpu_generator, device)
    encoded = _one_hot(labels.to(device), label_dim, device)
    features = generator(noise, encoded).cpu()
    return features, labels.cpu()


def sample_cgan_like_train(generator, y_train, latent_dim, label_dim, seed, device=None):
    """Draw one synthetic row per training row, conditioned on that row's own label."""
    labels = torch.as_tensor(y_train).long().view(-1)
    return sample_cgan(generator, labels, latent_dim, label_dim, seed, device=device)


def detection_auc(real_x, synth_x, n_splits=4, random_state=42, n_estimators=100, max_depth=None):
    """Balanced real-vs-synthetic forest on the exact rows it is given.

    Folds partition those rows, so a record is not in both the forest fit and the
    fold score. This is the checkpoint-selection detector. Final reporting uses
    `heldout_detection` instead.
    """
    real = real_x.detach().cpu().numpy() if torch.is_tensor(real_x) else np.asarray(real_x, dtype=float)
    synth = synth_x.detach().cpu().numpy() if torch.is_tensor(synth_x) else np.asarray(synth_x, dtype=float)
    if real.ndim != 2 or synth.ndim != 2 or real.shape[1] != synth.shape[1]:
        raise ValueError("Real and synthetic matrices must be 2D and the same width.")
    if len(real) != len(synth):
        raise ValueError("Real and synthetic matrices must be the same length.")
    if len(real) < n_splits * 2:
        raise ValueError("Need more rows per class for the stratified folds.")
    features = np.vstack([real, synth])
    labels = np.concatenate([np.zeros(len(real)), np.ones(len(synth))])
    folder = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for train_index, test_index in folder.split(features, labels):
        overlap = set(train_index.tolist()) & set(test_index.tolist())
        if overlap:
            raise RuntimeError("A detection fold reused rows.")
        forest = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=1,
        )
        forest.fit(features[train_index], labels[train_index])
        proba = forest.predict_proba(features[test_index])[:, 1]
        scores.append(float(roc_auc_score(labels[test_index], proba)))
    return {"fold_auc": scores, "mean_auc": float(np.mean(scores))}


def heldout_detection(real_common, synth_common, random_state=124, n_estimators=200, repeats=4, test_size=0.5):
    """Final detection AUC. Do not use the return value to pick a checkpoint or a seed.

    The split is a repeated stratified holdout, not the 4-fold CV used at checkpoints,
    and the caller is expected to pass held-out real rows plus a fresh synthetic draw.
    """
    real = _as_numpy(real_common)
    synth = _as_numpy(synth_common)
    if len(real) != len(synth) or real.shape[1] != synth.shape[1]:
        raise ValueError("Final detection needs matched real and synthetic matrices.")
    features = np.vstack([real, synth])
    labels = np.concatenate([np.zeros(len(real)), np.ones(len(synth))])
    scores = []
    for repeat in range(int(repeats)):
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state + repeat,
        )
        train_index, test_index = next(splitter.split(features, labels))
        if set(train_index.tolist()) & set(test_index.tolist()):
            raise RuntimeError("A final detection split reused rows.")
        forest = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=1)
        forest.fit(features[train_index], labels[train_index])
        proba = forest.predict_proba(features[test_index])[:, 1]
        scores.append(float(roc_auc_score(labels[test_index], proba)))
    std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    return {"repeat_auc": scores, "mean_auc": float(np.mean(scores)), "std_auc": std}


def save_generator(run, path):
    """Write generator weights next to the project. The path should stay gitignored."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": run["generator"].state_dict(),
        "config": run["config"],
        "kind": run["kind"],
        "seed": run["seed"],
        "selected_epoch": run.get("selected_epoch"),
        "acceptable": run.get("acceptable"),
        "label_dim": run.get("label_dim"),
        "checkpoints": run.get("checkpoints"),
        "selection": run.get("selection"),
        "d_loss": run.get("d_loss"),
        "g_loss": run.get("g_loss"),
    }, path)
    return path


def unconditional_efficacy():
    return {
        "available": False,
        "reason": (
            "The unconditional generator does not emit an income label. "
            "Copying training labels onto its rows is not a labeled-data protocol, "
            "so efficacy is omitted for the GAN."
        ),
    }


def efficacy_ratio(x_train, y_train, x_test, y_test, x_synth, y_synth, random_state=42, n_estimators=100):
    """Random forests on real vs. labeled synthetic rows, both scored on held-out real test rows."""
    if y_synth is None:
        return unconditional_efficacy()

    def as_numpy(value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    x_train = as_numpy(x_train)
    y_train = as_numpy(y_train).reshape(-1)
    x_test = as_numpy(x_test)
    y_test = as_numpy(y_test).reshape(-1)
    x_synth = as_numpy(x_synth)
    y_synth = as_numpy(y_synth).reshape(-1)
    forest_real = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    forest_synth = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    forest_real.fit(x_train, y_train)
    forest_synth.fit(x_synth, y_synth)
    auc_real = float(roc_auc_score(y_test, forest_real.predict_proba(x_test)[:, 1]))
    auc_synth = float(roc_auc_score(y_test, forest_synth.predict_proba(x_test)[:, 1]))
    return {
        "available": True,
        "auc_real": auc_real,
        "auc_synth": auc_synth,
        "ratio": auc_synth / auc_real,
    }


def search_grid(epoch_choices, output_activation):
    """The original-style screen, as configuration data. It does not train."""
    configs = []
    for latent_dim in (50, 64):
        for batch_size in (32, 64):
            for dropout in (0.0, 0.3):
                for weight_decay in (0.0, 1e-4):
                    for optimizer in ("Adam", "SGD", "AdamW"):
                        for lr in (1e-4, 1e-3):
                            for hidden_dim in (64, 128):
                                for epochs in epoch_choices:
                                    configs.append({
                                        "epochs": epochs,
                                        "latent_dim": latent_dim,
                                        "batch_size": batch_size,
                                        "lr": lr,
                                        "hidden_dim": hidden_dim,
                                        "dropout": dropout,
                                        "weight_decay": weight_decay,
                                        "optimizer": optimizer,
                                        "output_activation": output_activation,
                                    })
    return configs


if __name__ == "__main__":
    print(
        "Import this module from experiments.ipynb. "
        f"Place the supplied adult.arff at {DEFAULT_PATH}. "
        "Direct execution does not train."
    )

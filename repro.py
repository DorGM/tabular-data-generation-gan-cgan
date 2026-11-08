"""Read-only reproducibility helpers for the Adult GAN baseline.

Importing this module does not read the ARFF, load a checkpoint, or train.
The original training code stays in main.py.
"""

import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from main import decode_rows

PRIMARY_CONFIG = {
    "epochs": 120,
    "latent_dim": 64,
    "batch_size": 64,
    "lr": 1e-3,
    "hidden_dim": 128,
    "dropout": 0.0,
    "weight_decay": 0.0,
    "optimizer": "Adam",
    "output_activation": None,
}
FOLLOWUP_CONFIG = dict(PRIMARY_CONFIG, lr=1e-4)
SPLIT_SEEDS = (42, 68, 124)
PROJECT_DIR = Path(__file__).resolve().parent


def config_fingerprint(config):
    """Stable hash of a JSON-serializable configuration. Key order does not matter."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def environment_report():
    """Versions of the libraries used to read the baseline. This does not retrain."""
    import pandas
    import scipy
    import sklearn
    import torch

    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
    }
    try:
        import matplotlib

        report["matplotlib"] = matplotlib.__version__
    except ImportError:
        report["matplotlib"] = None
    return report


def decoded_selection_age_mean(bundle):
    """Mean age of every selection row after inverse scaling.

    The first rows of the slice are not a substitute for this mean.
    """
    rows = int(bundle["x_select"].shape[0])
    decoded = decode_rows(
        bundle["x_select"],
        bundle["feature_names"],
        bundle["numeric_columns"],
        bundle["categorical_groups"],
        bundle["scaler"],
    )
    if decoded["numeric"].shape[0] != rows:
        raise RuntimeError("Selection age mean did not use every selection row.")
    return float(decoded["numeric"][:, 0].mean())


def summarize_saved_run(path):
    """Metadata from a summary checkpoint. Weights are not returned and the file is not written."""
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    selected = obj.get("selected_epoch")
    return {
        "file": Path(path).name,
        "kind": obj.get("kind"),
        "seed": obj.get("seed"),
        "acceptable": obj.get("acceptable"),
        "selected_epoch_index": selected,
        "selected_epoch_number": None if selected is None else int(selected) + 1,
        "config_fingerprint": config_fingerprint(obj.get("config") or {}),
        "config": obj.get("config"),
        "n_checkpoints": len(obj.get("checkpoints") or []),
    }

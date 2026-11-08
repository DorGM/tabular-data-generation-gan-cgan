"""Score the original lr 1e-3 checkpoints on the selection slice.

Does not train, does not write into checkpoints/, and does not read the test
split for this report. Ineligible files are last-epoch weights, not selected models.
"""

import json
from pathlib import Path

import numpy as np
import torch

from evaluator import (
    fidelity_report,
    frame_from_legacy_decoded,
    hard_validity,
    privacy_report,
    schema_from_frame,
    selection_detection,
)
from main import (
    ConditionalGenerator,
    Generator,
    decode_rows,
    load_adult,
    prepare_experiment,
    sample_cgan,
    sample_gan,
)

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
OUT_DIR = ROOT / "results" / "stage1"
NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = [
    "workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country",
]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None:
        return None
    return value


def _load_generator(record, width):
    config = record["config"]
    if record["kind"] == "gan":
        generator = Generator(
            int(config["latent_dim"]), width, hidden_dim=int(config["hidden_dim"]),
            dropout=float(config["dropout"]), output_activation=config["output_activation"],
        )
    elif record["kind"] == "cgan":
        generator = ConditionalGenerator(
            int(config["latent_dim"]), 2, width, hidden_dim=int(config["hidden_dim"]),
            dropout=float(config["dropout"]), output_activation=config["output_activation"],
        )
    else:
        raise ValueError(record["kind"])
    generator.load_state_dict(record["state_dict"])
    generator.eval()
    return generator


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_adult()
    schema = schema_from_frame(frame, NUMERIC, CATEGORICAL)
    bundles = {}
    rows = []
    paths = sorted(path for path in CHECKPOINT_DIR.glob("*.pt") if "lr1e-4" not in path.name)
    if len(paths) != 12:
        raise RuntimeError(f"Expected 12 primary checkpoints, found {len(paths)}.")
    for path in paths:
        record = torch.load(path, map_location="cpu", weights_only=False)
        config = record["config"]
        if float(config["lr"]) != 1e-3:
            raise RuntimeError(f"{path.name} is not the primary learning rate.")
        seed = int(record["seed"])
        activation = config["output_activation"]
        key = (seed, activation)
        if key not in bundles:
            bundles[key] = prepare_experiment(frame, seed=seed, output_activation=activation)
        bundle = bundles[key]
        real_decoded = decode_rows(
            bundle["x_select"], bundle["feature_names"], bundle["numeric_columns"],
            bundle["categorical_groups"], bundle["scaler"],
        )
        real_frame = frame_from_legacy_decoded(real_decoded, bundle, income=bundle["y_select"].numpy())
        real_validity = hard_validity(real_frame, schema)
        if not real_validity["valid"]:
            raise RuntimeError(f"Real selection rows failed hard validity for seed {seed} {activation}: {real_validity}")
        generator = _load_generator(record, bundle["x_train"].shape[1])
        if record["kind"] == "gan":
            synthetic = sample_gan(generator, bundle["x_select"].shape[0], int(config["latent_dim"]), 42, device="cpu")
            synth_frame = frame_from_legacy_decoded(
                decode_rows(synthetic, bundle["feature_names"], bundle["numeric_columns"], bundle["categorical_groups"], bundle["scaler"]),
                bundle,
            )
            income = None
        else:
            synthetic, labels = sample_cgan(
                generator, bundle["y_select"], int(config["latent_dim"]), 2, 42, device="cpu",
            )
            synth_frame = frame_from_legacy_decoded(
                decode_rows(synthetic, bundle["feature_names"], bundle["numeric_columns"], bundle["categorical_groups"], bundle["scaler"]),
                bundle,
                income=labels.numpy(),
            )
            income = "selection row labels"
        fit_decoded = decode_rows(
            bundle["x_train"], bundle["feature_names"], bundle["numeric_columns"],
            bundle["categorical_groups"], bundle["scaler"],
        )
        fit_frame = frame_from_legacy_decoded(fit_decoded, bundle)
        synthetic_validity = hard_validity(synth_frame.drop(columns=["income"], errors="ignore"), schema)
        if record["kind"] == "cgan":
            distances = fidelity_report(real_frame, synth_frame, schema)
        else:
            distances = fidelity_report(
                real_frame.drop(columns=["income"]),
                synth_frame,
                schema,
            )
        privacy = privacy_report(synth_frame, fit_frame, schema)
        detection = selection_detection(real_frame, synth_frame, schema)
        selected = record.get("selected_epoch")
        result = {
            "file": path.name,
            "kind": record["kind"],
            "seed": seed,
            "output_activation": activation,
            "lr": float(config["lr"]),
            "acceptable": bool(record.get("acceptable")),
            "weights": "selected_epoch" if record.get("acceptable") else "last_epoch_not_a_selected_model",
            "selected_epoch_number": None if selected is None else int(selected) + 1,
            "sample_seed": 42,
            "split": "selection",
            "n_selection": int(bundle["x_select"].shape[0]),
            "uses_final_test": False,
            "conditioning": income,
            "real_selection_violation_total": real_validity["violation_total"],
            "synthetic_hard_validity": synthetic_validity,
            "fidelity_summary": {
                "numeric_ks_mean": distances["numeric_ks_mean"],
                "numeric_wasserstein_over_std_mean": distances["numeric_wasserstein_over_std_mean"],
                "categorical_tv_mean": distances["categorical_tv_mean"],
                "category_coverage_mean": distances["category_coverage_mean"],
                "rare_category_recall": distances["rare_category_recall"],
                "rare_category_count": distances["rare_category_count"],
                "correlation_mae": distances["correlation_mae"],
                "pairwise_categorical_tv_mean": distances["pairwise_categorical_tv_mean"],
                "synthetic_unique_fraction": distances["synthetic_unique_fraction"],
                "synthetic_duplicate_rate": distances["synthetic_duplicate_rate"],
                "nearest_neighbor": distances["nearest_neighbor"],
                "numeric": distances["numeric"],
                "categorical": distances["categorical"],
                "label_conditional": distances["label_conditional"],
            },
            "privacy": privacy,
            "selection_detection": detection,
        }
        rows.append(_jsonable(result))
        (OUT_DIR / "legacy_selection_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(
            path.name,
            "valid", synthetic_validity["valid"],
            "violations", synthetic_validity["violation_total"],
            "ks", None if distances["numeric_ks_mean"] is None else round(distances["numeric_ks_mean"], 4),
            "tv", None if distances["categorical_tv_mean"] is None else round(distances["categorical_tv_mean"], 4),
            "auc", round(detection["mean_auc"], 4),
            "copies", round(privacy["exact_duplicate_fraction_vs_reference"], 4),
            flush=True,
        )
    print("wrote", OUT_DIR / "legacy_selection_metrics.json", "runs", len(rows), flush=True)


if __name__ == "__main__":
    main()

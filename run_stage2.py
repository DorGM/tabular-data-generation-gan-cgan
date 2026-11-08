"""Train the type-aware cGAN on the original sigmoid splits.

Selection-only. The test split is not read. Checkpoints go under results/stage2
and do not replace the legacy files in checkpoints/.

Comparison checkpoint, predeclared before the run and used only for this experiment:
lowest hard-validity violation fraction, then lowest mean of KS and categorical
TV, then lowest protocol A AUC, then the earlier epoch. Epoch 120 is always
reported as well.
"""

import json
from pathlib import Path

import numpy as np
import torch

from evaluator import (
    fidelity_report,
    frame_from_legacy_decoded,
    hard_validity,
    schema_from_frame,
    selection_detection,
)
from main import decode_rows, load_adult, prepare_experiment
from repro import PRIMARY_CONFIG, config_fingerprint
from type_aware import TypeAwareConditionalGenerator, fit_type_spec, train_type_aware_cgan

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "stage2"
NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = [
    "workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country",
]
SEEDS = (42, 68, 124)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None:
        return None
    return value


def _one_hot(labels, device):
    index = torch.as_tensor(labels, device=device).long().view(-1, 1)
    encoded = torch.zeros(index.shape[0], 2, device=device)
    encoded.scatter_(1, index, 1.0)
    return encoded


def score_checkpoint(generator, bundle, schema, device):
    generator.eval()
    n_rows = int(bundle["x_select"].shape[0])
    cpu_generator = torch.Generator()
    cpu_generator.manual_seed(42)
    noise = torch.randn(n_rows, generator.latent_dim, generator=cpu_generator).to(device)
    labels = _one_hot(bundle["y_select"], device)
    encoded = generator(noise, labels, temperature=0.5, hard=True).detach().cpu()
    synthetic = frame_from_legacy_decoded(
        decode_rows(
            encoded, bundle["feature_names"], bundle["numeric_columns"],
            bundle["categorical_groups"], bundle["scaler"],
        ),
        bundle,
        income=bundle["y_select"].numpy(),
    )
    real = frame_from_legacy_decoded(
        decode_rows(
            bundle["x_select"], bundle["feature_names"], bundle["numeric_columns"],
            bundle["categorical_groups"], bundle["scaler"],
        ),
        bundle,
        income=bundle["y_select"].numpy(),
    )
    validity = hard_validity(synthetic.drop(columns=["income"]), schema)
    distances = fidelity_report(real, synthetic, schema)
    detection = selection_detection(real.drop(columns=["income"]), synthetic.drop(columns=["income"]), schema)
    return {
        "hard_validity": validity,
        "numeric_ks_mean": distances["numeric_ks_mean"],
        "categorical_tv_mean": distances["categorical_tv_mean"],
        "category_coverage_mean": distances["category_coverage_mean"],
        "rare_category_recall": distances["rare_category_recall"],
        "correlation_mae": distances["correlation_mae"],
        "synthetic_duplicate_rate": distances["synthetic_duplicate_rate"],
        "selection_detection_auc": detection["mean_auc"],
        "selection_detection_oriented_auc": detection["oriented_auc"],
        "uses_final_test": False,
    }


def comparison_key(row):
    fraction = row["hard_validity"]["row_violation_fraction"]
    ks = row["numeric_ks_mean"] if row["numeric_ks_mean"] is not None else 1.0
    tv = row["categorical_tv_mean"] if row["categorical_tv_mean"] is not None else 1.0
    auc = row["selection_detection_auc"]
    return (fraction, 0.5 * ks + 0.5 * tv, auc, row["epoch_number"])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = OUT_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    frame = load_adult()
    schema = schema_from_frame(frame, NUMERIC, CATEGORICAL)
    config = dict(PRIMARY_CONFIG, output_activation="type_aware_minmax_encoded")
    history_path = OUT_DIR / "history.jsonl"
    device = torch.device("cpu")
    for seed in SEEDS:
        bundle = prepare_experiment(frame, seed=seed, output_activation="sigmoid")
        spec = fit_type_spec(frame, bundle)
        print(f"training type-aware cgan seed {seed}", flush=True)
        resume_path = OUT_DIR / f"resume_seed{seed}.pt"
        resume = None
        if resume_path.exists():
            resume = torch.load(resume_path, map_location="cpu", weights_only=False)
            print(f"  resuming at epoch {int(resume['next_epoch']) + 1}", flush=True)

        def on_epoch(epoch, generator, temperature, d_loss, g_loss, seed=seed, bundle=bundle, spec=spec):
            epoch_number = epoch + 1
            scored = score_checkpoint(generator, bundle, schema, device)
            path = checkpoint_dir / f"type_aware_cgan_seed{seed}_epoch{epoch_number:04d}.pt"
            torch.save(
                {
                    "state_dict": {key: value.detach().cpu() for key, value in generator.state_dict().items()},
                    "config": config,
                    "seed": seed,
                    "epoch_number": epoch_number,
                    "temperature": temperature,
                    "kind": "type_aware_cgan",
                    "education_levels": spec["education_levels"],
                },
                path,
            )
            row = {
                "seed": seed,
                "epoch_number": epoch_number,
                "temperature": temperature,
                "d_loss": d_loss,
                "g_loss": g_loss,
                "checkpoint": path.name,
                "config_fingerprint": config_fingerprint(config),
                **scored,
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_jsonable(row)) + "\n")
            print(
                f"  seed {seed} epoch {epoch_number} violations {scored['hard_validity']['violation_total']} "
                f"ks {scored['numeric_ks_mean']:.4f} tv {scored['categorical_tv_mean']:.4f} "
                f"auc {scored['selection_detection_auc']:.4f}",
                flush=True,
            )

        train_type_aware_cgan(
            bundle["x_train"], bundle["y_train"], spec, config,
            seed=seed, device=device, log_every=10, on_epoch=on_epoch,
            resume=resume, resume_path=resume_path,
        )
    rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    chosen = {}
    for seed in SEEDS:
        seed_rows = [row for row in rows if row["seed"] == seed]
        if not seed_rows:
            chosen[str(seed)] = None
            continue
        best = min(seed_rows, key=comparison_key)
        epoch_120 = next(row for row in seed_rows if row["epoch_number"] == 120) if any(
            row["epoch_number"] == 120 for row in seed_rows
        ) else None
        chosen[str(seed)] = {"comparison_checkpoint": best, "epoch_120": epoch_120}
    summary = {
        "rule": "violation fraction, then mean of KS and TV, then protocol A AUC, then earlier epoch",
        "uses_final_test": False,
        "seeds": chosen,
    }
    (OUT_DIR / "comparison.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print("wrote", OUT_DIR / "comparison.json", flush=True)


if __name__ == "__main__":
    main()

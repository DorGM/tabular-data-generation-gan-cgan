"""Reference generators from SDV, scored with the same protocol as ladder.py.

Gaussian copula, CTGAN and TVAE are fit on the fit slice with default settings
(CTGAN and TVAE: 300 epochs). There is no checkpoint selection. Income is a
generated column, so utility uses the synthetic income labels (train on
synthetic, test on real). Importing this module does not read data or train.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer

from evaluator import _design_matrix, final_detection, schema_from_frame, utility_report
from ladder import CATEGORICAL, NUMERIC, OUT_DIR, SEEDS, score, split_frames
from main import load_adult

MODELS = {
    "sdv_gaussian_copula": lambda md, seed: GaussianCopulaSynthesizer(md),
    "sdv_ctgan": lambda md, seed: CTGANSynthesizer(md, epochs=300, verbose=False),
    "sdv_tvae": lambda md, seed: TVAESynthesizer(md, epochs=300),
}


def run(name, seed, frame, schema):
    fit, select, test = split_frames(frame, seed)
    table = fit.copy()
    table["income"] = table["income"].astype(str)
    metadata = Metadata.detect_from_dataframes({"adult": table})
    for column in CATEGORICAL + ["income"]:
        metadata.update_column(column_name=column, sdtype="categorical", table_name="adult")
    for column in NUMERIC:
        metadata.update_column(column_name=column, sdtype="numerical", table_name="adult")
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    model = MODELS[name](metadata, seed)
    model.fit({"adult": table})

    def draw(n):
        rows = model.sample(n)
        if isinstance(rows, dict):
            rows = next(iter(rows.values()))
        rows["income"] = rows["income"].astype(int)
        return rows.reset_index(drop=True)

    select_features = select.drop(columns=["income"])
    dev = draw(len(select))
    test_synth = draw(len(test))
    utility_synth = draw(len(fit))
    selection = score(select_features, dev[NUMERIC + CATEGORICAL], schema)
    final = final_detection(select_features, dev[NUMERIC + CATEGORICAL], test.drop(columns=["income"]),
                            test_synth[NUMERIC + CATEGORICAL], schema)
    utility = utility_report(
        _design_matrix(fit.drop(columns=["income"]), schema), fit["income"].to_numpy(),
        _design_matrix(utility_synth[NUMERIC + CATEGORICAL], schema), utility_synth["income"].to_numpy(),
        _design_matrix(test.drop(columns=["income"]), schema), test["income"].to_numpy(),
    )
    test_fidelity = score(test.drop(columns=["income"]), test_synth[NUMERIC + CATEGORICAL], schema)
    return {
        "rung": name, "seed": seed, "selected": selection,
        "final_detection_auc": final["mean_auc"], "utility_ratio": utility["ratio"],
        "utility_auc_real": utility["auc_real"], "utility_auc_synthetic": utility["auc_synthetic"],
        "test_ks": test_fidelity["ks"], "test_tv": test_fidelity["tv"],
        "test_coverage": test_fidelity["coverage"], "test_violations": test_fidelity["violations"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_adult()
    schema = schema_from_frame(frame, NUMERIC, CATEGORICAL)
    for name in args.models:
        for seed in args.seeds:
            out = OUT_DIR / f"{name}_seed{seed}.json"
            if out.exists():
                print("skip", out.name, flush=True)
                continue
            result = run(name, seed, frame, schema)
            out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
            print(f"DONE {name} seed {seed}: final AUC {result['final_detection_auc']:.4f} "
                  f"utility {result['utility_ratio']:.3f}", flush=True)


if __name__ == "__main__":
    main()

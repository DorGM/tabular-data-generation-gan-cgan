"""Score every generator on a fresh holdout: the official Adult test file.

The ladder's own test split was used to study which rows the detector catches,
so it is no longer untouched. This script scores all models on the 16,281 rows
of the official `adult.test` file, which no model, rule or threshold has seen.
Training data, the fit slice and the selection slice are unchanged.

For each model and split seed it reports:
  - detection AUC with two detectors (random forest and gradient boosting),
    each fit on selection rows vs. a synthetic draw and scored on holdout rows
    vs. a fresh synthetic draw;
  - train-on-synthetic / test-on-real utility on the holdout;
  - KS, TV, coverage, hard-validity and domain-rule violations;
  - a memorization check: is each synthetic row closer to the training rows
    than to holdout rows? About 50% means no memorization.

Variant 6 (rung 5 plus snapping capital values to common real values) is built
here. The minimum count k for a value to be kept is chosen on the selection
slice only, from {2, 3, 5, 10, 20, 50}. k = 1 is excluded up front because a
value seen once belongs to one person.

Diffusion steps 7-9 (diffusion.py) are scored the same way. Step 10 is built here:
rows from step 5's GAN are noised to a level t0 and denoised by step 8's diffusion
model, then passed through the rule filter. t0 is chosen on the selection slice only,
from REFINE_LEVELS (out of 1,000 noise steps). Sampling time per 1,000 rows is reported
for every generator.

Importing this module does not read data or train.
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from evaluator import _design_matrix, final_detection, schema_from_frame, selection_detection, utility_report
from ladder import CATEGORICAL, MODEL_DIR, NUMERIC, SEEDS, load_generator, rule_violations, score, split_frames
from main import INCOME_MAP, NUMERIC_COLUMNS, load_adult

ROOT = Path(__file__).resolve().parent
HOLDOUT_PATH = ROOT / "data" / "adult_test.csv"
OUT_DIR = ROOT / "results" / "holdout"
SDV_DIR = ROOT / "models" / "sdv"
CAPITAL = ["capital-gain", "capital-loss"]
SNAP_CANDIDATES = (2, 3, 5, 10, 20, 50)
REFINE_LEVELS = (25, 50, 100, 200, 400)
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
           "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
           "income"]
MODELS = ["real_holdout", "0_baseline", "1_sampling", "2_st_onehot", "3_wgan_gp", "4_ctgan_norm",
          "4_ctgan_norm+reject", "5_rules", "6_rules_snap",
          "7_ddpm", "7_ddpm+reject", "8_ddpm_rules", "9_latent_rules", "10_gan_refine",
          "sdv_gaussian_copula", "sdv_tvae", "sdv_ctgan", "sdv_ctgan+reject"]


def load_holdout(path=HOLDOUT_PATH):
    """The official adult.test rows (16,281). Accepts the UCI file or a plain CSV of the same rows."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Put the official Adult test file (adult.test from UCI) at {path}.")
    first = path.read_text(encoding="utf-8").splitlines()[0]
    frame = pd.read_csv(path, header=None, names=COLUMNS, skipinitialspace=True,
                        skiprows=1 if first.startswith("|") else 0)
    frame = frame.dropna(subset=["income"])
    frame["income"] = frame["income"].str.rstrip(".").map(INCOME_MAP).astype(int)
    for column in NUMERIC_COLUMNS:
        frame[column] = frame[column].astype(float)
    return frame.drop(columns=["education"]).reset_index(drop=True)


# ------------------------------------------------------------ helpers ----

def snap_capital(frame, value_sets):
    """Move each non-zero capital value to the nearest value in its kept set."""
    frame = frame.copy()
    for column in CAPITAL:
        kept = value_sets[column]
        values = frame[column].to_numpy(dtype=float)
        nonzero = values > 0
        position = np.clip(np.searchsorted(kept, values[nonzero]), 1, len(kept) - 1)
        left, right = kept[position - 1], kept[position]
        values[nonzero] = np.where(values[nonzero] - left <= right - values[nonzero], left, right)
        frame[column] = values
    return frame


def value_sets_for(fit, k):
    sets = {}
    for column in CAPITAL:
        counts = fit[column][fit[column] > 0].value_counts()
        sets[column] = np.sort(counts[counts >= k].index.to_numpy(dtype=float))
    return sets


def choose_snap_threshold(sample, fit, select, schema):
    """Pick k on the selection slice: lowest cross-validated detection AUC, ties go to the larger k."""
    draw = sample(select["income"].to_numpy(), seed=42)
    scores = {}
    for k in SNAP_CANDIDATES:
        snapped = snap_capital(draw, value_sets_for(fit, k))
        scores[k] = selection_detection(select.drop(columns=["income"]), snapped[NUMERIC + CATEGORICAL],
                                        schema)["mean_auc"]
    best = min(SNAP_CANDIDATES, key=lambda k: (round(scores[k], 4), -k))
    return best, scores


def boosting_detection(dev_real, dev_synth, test_real, test_synth, schema):
    x_train = np.vstack([_design_matrix(dev_real, schema), _design_matrix(dev_synth, schema)])
    y_train = np.r_[np.zeros(len(dev_real)), np.ones(len(dev_synth))]
    x_test = np.vstack([_design_matrix(test_real, schema), _design_matrix(test_synth, schema)])
    y_test = np.r_[np.zeros(len(test_real)), np.ones(len(test_synth))]
    model = HistGradientBoostingClassifier(random_state=124).fit(x_train, y_train)
    return float(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]))


def memorization(synth, fit, holdout, schema, seed, n_ref=10000, n_synth=5000):
    """Distance from each synthetic row to its nearest training row and nearest holdout row."""
    rng = np.random.default_rng(seed)
    pick = lambda frame, n: frame.iloc[rng.choice(len(frame), size=min(n, len(frame)), replace=False)]
    fit_ref, hold_ref, synth = pick(fit, n_ref), pick(holdout, n_ref), pick(synth, n_synth)
    base = _design_matrix(fit, schema)
    center, spread = base.mean(0), base.std(0)
    spread[spread < 1e-9] = 1.0
    scale = lambda frame: (_design_matrix(frame, schema) - center) / spread
    to_fit = NearestNeighbors(n_neighbors=1).fit(scale(fit_ref)).kneighbors(scale(synth))[0][:, 0]
    to_hold = NearestNeighbors(n_neighbors=1).fit(scale(hold_ref)).kneighbors(scale(synth))[0][:, 0]
    closer = np.where(np.isclose(to_fit, to_hold), 0.5, (to_fit < to_hold).astype(float))
    return {
        "dcr_fit_median": float(np.median(to_fit)), "dcr_holdout_median": float(np.median(to_hold)),
        "closer_to_fit_share": float(closer.mean()), "exact_copy_share": float((to_fit < 1e-9).mean()),
    }


# ------------------------------------------------------------- models ----

def refine_sampler(seed, select, schema):
    """Step 10: step 5's GAN rows, noised to t0 and denoised by step 8. t0 is chosen on the selection slice."""
    import torch
    from diffusion import Sampler

    gan = load_generator(MODEL_DIR / f"5_rules_seed{seed}.pt", reject=False)
    sampler = Sampler(torch.load(MODEL_DIR / f"8_ddpm_rules_seed{seed}.pt", map_location="cpu", weights_only=False))
    model, encoder = sampler.model, sampler.encoder
    stride = model.T / sampler.config["sample_steps"]

    def draw_at(t0):
        @torch.no_grad()
        def draw(labels, draw_seed):
            rows = gan(labels, seed=draw_seed)[NUMERIC + CATEGORICAL]
            x_num, x_cat = encoder.transform(rows)
            g = torch.Generator().manual_seed(int(draw_seed) + 1)
            y = torch.as_tensor(labels, dtype=torch.long)
            t = torch.full((len(labels),), int(t0), dtype=torch.long)
            noisy = model.q_sample(torch.tensor(x_num), torch.tensor(x_cat), t, g)[:2]
            out_num, out_cat = model.sample(y, max(1, round(t0 / stride)), g, start=noisy, t_start=t0)
            return encoder.inverse(out_num.numpy(), out_cat.numpy())
        return draw

    select_x = select.drop(columns=["income"])
    scores = {}
    for t0 in REFINE_LEVELS:
        rows = sampler(select["income"].to_numpy(), seed=42, reject=True, draw=draw_at(t0))
        scores[t0] = selection_detection(select_x, rows[NUMERIC + CATEGORICAL], schema)["mean_auc"]
    best = min(REFINE_LEVELS, key=lambda t0: (round(scores[t0], 4), t0))
    sample = lambda labels, seed=0, stats=None: sampler(labels, seed, stats, reject=True, draw=draw_at(best))
    return sample, {"refine_t0": best, "refine_selection_auc": scores}


def sdv_sampler(name, seed, fit):
    """Fit (or reload) an SDV synthesizer on the fit slice. Returns a function n -> frame with income."""
    from sdv.metadata import Metadata
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer
    import torch

    base = name.replace("+reject", "")
    path = SDV_DIR / f"{base}_seed{seed}.pkl"
    if path.exists():
        model = pickle.loads(path.read_bytes())
    else:
        table = fit.copy()
        table["income"] = table["income"].astype(str)
        metadata = Metadata.detect_from_dataframes({"adult": table})
        for column in CATEGORICAL + ["income"]:
            metadata.update_column(column_name=column, sdtype="categorical", table_name="adult")
        for column in NUMERIC:
            metadata.update_column(column_name=column, sdtype="numerical", table_name="adult")
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = {"sdv_gaussian_copula": lambda: GaussianCopulaSynthesizer(metadata),
                 "sdv_ctgan": lambda: CTGANSynthesizer(metadata, epochs=300, verbose=False),
                 "sdv_tvae": lambda: TVAESynthesizer(metadata, epochs=300)}[base]()
        model.fit({"adult": table})
        SDV_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(model))

    def draw_once(n):
        rows = model.sample(n)
        if isinstance(rows, dict):
            rows = next(iter(rows.values()))
        rows["income"] = rows["income"].astype(int)
        return rows.reset_index(drop=True)

    def sample(n, stats=None):
        rows = draw_once(n)
        if not name.endswith("+reject"):
            return rows
        bad = rule_violations(rows)[0]
        if stats is not None:
            stats["rejected_share"] = float(bad.mean())
        rows = rows[~bad]
        while len(rows) < n:
            extra = draw_once(n)
            rows = pd.concat([rows, extra[~rule_violations(extra)[0]]], ignore_index=True)
        return rows.iloc[:n].reset_index(drop=True)

    return sample


def evaluate(name, seed, frame, holdout, schema):
    fit, select, _old_test = split_frames(frame, seed)
    features = NUMERIC + CATEGORICAL
    select_x = select.drop(columns=["income"])
    stats, extra = {}, {}

    if name == "real_holdout":
        # Real rows the models never trained on (the ladder's old test split) stand in for a generator.
        # This is the best score a generator could hope for.
        rng = np.random.default_rng(seed)
        reference = _old_test.iloc[rng.permutation(len(_old_test))].reset_index(drop=True)
        half = len(reference) // 2
        dev, test_synth = reference.iloc[:half], reference.iloc[half:]
        dev_real = select.iloc[rng.choice(len(select), half, replace=False)].drop(columns=["income"])
        test_real = holdout.iloc[rng.choice(len(holdout), len(test_synth), replace=False)]
        utility_synth = reference
        utility_real = fit.iloc[rng.choice(len(fit), len(reference), replace=False)]
    else:
        dev_real, test_real, utility_real = select_x, holdout, fit
        if name.startswith("sdv_"):
            sample = sdv_sampler(name, seed, fit)
            dev = sample(len(select))
            test_synth = sample(len(holdout), stats)
            utility_synth = sample(len(fit))
        elif name == "10_gan_refine" or name.split("_")[0] in {"7", "8", "9"}:
            from diffusion import load_diffusion

            if name == "10_gan_refine":
                generator, extra = refine_sampler(seed, select, schema)
            else:
                step = name.replace("+reject", "")
                generator = load_diffusion(MODEL_DIR / f"{step}_seed{seed}.pt",
                                           reject=True if name.endswith("+reject") else None)
            dev = generator(select["income"].to_numpy(), seed=7)
            started = time.time()
            test_synth = generator(holdout["income"].to_numpy(), seed=9000 + seed, stats=stats)
            stats["sample_seconds_per_1k"] = 1000 * (time.time() - started) / len(holdout)
            utility_synth = generator(fit["income"].to_numpy(), seed=11)
        else:
            rung = {"4_ctgan_norm+reject": "4_ctgan_norm", "6_rules_snap": "5_rules"}.get(name, name)
            reject = True if name == "4_ctgan_norm+reject" else None
            generator = load_generator(MODEL_DIR / f"{rung}_seed{seed}.pt", reject=reject)
            post = lambda rows: rows
            if name == "6_rules_snap":
                k, k_scores = choose_snap_threshold(generator, fit, select, schema)
                sets = value_sets_for(fit, k)
                post = lambda rows: snap_capital(rows, sets)
                extra = {"snap_k": k, "snap_selection_auc": k_scores,
                         "snap_values_kept": {c: int(len(v)) for c, v in sets.items()}}
            dev = post(generator(select["income"].to_numpy(), seed=7))
            started = time.time()
            test_synth = post(generator(holdout["income"].to_numpy(), seed=9000 + seed, stats=stats))
            stats["sample_seconds_per_1k"] = 1000 * (time.time() - started) / len(holdout)
            utility_synth = post(generator(fit["income"].to_numpy(), seed=11))

    test_real_x = test_real.drop(columns=["income"])
    rf = final_detection(dev_real, dev[features], test_real_x, test_synth[features], schema)["mean_auc"]
    gb = boosting_detection(dev_real, dev[features], test_real_x, test_synth[features], schema)
    utility = utility_report(
        _design_matrix(utility_real.drop(columns=["income"]), schema), utility_real["income"].to_numpy(),
        _design_matrix(utility_synth[features], schema), utility_synth["income"].to_numpy(),
        _design_matrix(holdout.drop(columns=["income"]), schema), holdout["income"].to_numpy(),
    )
    fidelity = score(test_real_x, test_synth[features], schema) if len(test_real_x) == len(test_synth) else None
    _mask, rules = rule_violations(test_synth)
    return {
        "model": name, "seed": seed, "n_holdout": int(len(holdout)),
        "detection_auc_rf": rf, "detection_auc_gb": gb,
        "utility_ratio": utility["ratio"], "utility_auc_real": utility["auc_real"],
        "utility_auc_synthetic": utility["auc_synthetic"],
        "ks": fidelity["ks"] if fidelity else None, "tv": fidelity["tv"] if fidelity else None,
        "coverage": fidelity["coverage"] if fidelity else None,
        "hard_violations": fidelity["violations"] if fidelity else None,
        "rule_violations": rules, "rule_violation_share": float(_mask.mean()),
        "rejected_share": stats.get("rejected_share"),
        "sample_seconds_per_1k": stats.get("sample_seconds_per_1k"),
        **memorization(test_synth[features], fit[features], holdout[features], schema, seed),
        **extra,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_adult()
    holdout = load_holdout()
    schema = schema_from_frame(frame, NUMERIC, CATEGORICAL)
    for name in args.models:
        for seed in args.seeds:
            out = OUT_DIR / f"{name}_seed{seed}.json"
            if out.exists():
                print("skip", out.name, flush=True)
                continue
            result = evaluate(name, seed, frame, holdout, schema)
            out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
            print(f"DONE {name} seed {seed}: RF {result['detection_auc_rf']:.4f} GB {result['detection_auc_gb']:.4f} "
                  f"utility {result['utility_ratio']:.3f} closer-to-fit {result['closer_to_fit_share']:.3f}",
                  flush=True)


if __name__ == "__main__":
    main()

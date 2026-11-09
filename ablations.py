"""Follow-up checks on the Adult benchmark, run on the saved checkpoints (no training).

The official Adult test file has now been looked at many times while the ladder was built,
so it is a reused benchmark here. Every choice below (noise level, decoding) is still made on
the selection slice only; the test-file numbers are reported for comparison with the main table.

  copies    Exact-copy check over every row generated during evaluation (all three draws of
            every model and seed) against every fit-slice row, instead of a sample.
  decoding  Step 9 (latent diffusion) loses rare categories. Is it the argmax decoder or the
            learned latent distribution? Compares argmax with sampling from the decoder's
            softmax, for generated latents and for real rows passed through the VAE.
  hybrid    Does step 10's gain come from starting at GAN rows? Compares, with the same step 8
            denoiser: GAN step 5 start, a weaker GAN (step 4) start, an independent-marginals
            start, and pure diffusion from noise with the same number of denoiser calls.

Usage: python ablations.py {copies,decoding,hybrid} [--seeds 42 68 124]
Results go to results/ablations/.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diffusion import Sampler
from evaluate_holdout import MODELS, draw_three, draws, load_holdout, memorization, score_draws
from evaluator import fidelity_report, schema_from_frame, selection_detection
from ladder import CATEGORICAL, MODEL_DIR, NUMERIC, SEEDS, load_generator, rule_violations, split_frames
from main import load_adult

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "ablations"
FEATURES = NUMERIC + CATEGORICAL
# Step 10's grid plus two higher levels: two of three seeds picked the old top level (400).
REFINE_LEVELS_EXT = (25, 50, 100, 200, 400, 600, 800)
PURE_STEPS = (20, 40, 60, 80)


def _write(name, payload):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / name).write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


# ------------------------------------------------------------------ copies ----

# fnlwgt is close to unique per person, so an exact match on all 13 columns is a narrow test.
# Matching on the other 12 columns is the quasi-identifier version.
NO_FNLWGT = [c for c in FEATURES if c != "fnlwgt"]


def _row_keys(frame, columns=None):
    """One hashable key per row; numbers rounded so 39 and 39.0 match (+0.0 turns -0.0 into 0.0)."""
    parts = []
    for column in columns or FEATURES:
        values = frame[column]
        if column in NUMERIC:
            parts.append((np.round(values.to_numpy(dtype=float), 6) + 0.0).astype(str))
        else:
            parts.append(values.astype(str).str.strip().to_numpy())
    return pd.Series(["\x1f".join(row) for row in zip(*parts)], index=frame.index)


def exact_copy_count(synth, real, columns=None):
    real_keys = set(_row_keys(real, columns))
    return int(_row_keys(synth, columns).isin(real_keys).sum())


def real_copy_rate(frame, seeds):
    """Scale for the copy check: how often real unseen rows (selection slice) equal a fit row."""
    rows = matches = matches_12 = 0
    for seed in seeds:
        fit, select, _ = split_frames(frame, seed)
        rows += len(select)
        matches += exact_copy_count(select, fit)
        matches_12 += exact_copy_count(select, fit, NO_FNLWGT)
    payload = {"rows": int(rows), "matches": int(matches), "matches_without_fnlwgt": int(matches_12),
               "seeds": list(seeds)}
    _write("real_copy_rate.json", payload)
    return payload


def run_copies(seeds):
    frame, holdout = load_adult(), load_holdout()
    print("real selection rows equal to a fit row:", real_copy_rate(frame, seeds), flush=True)
    schema = schema_from_frame(frame, NUMERIC, CATEGORICAL)
    out = {}
    for name in [m for m in MODELS if m != "real_holdout"]:
        for seed in seeds:
            d = draws(name, seed, frame, holdout, schema)
            fit = d["fit"]
            row = {"fit_rows": int(len(fit)), "all_train_rows": int(len(frame))}
            for key in ("dev", "test_synth", "utility_synth"):
                synth = d[key]
                row[f"{key}_rows"] = int(len(synth))
                row[f"{key}_copies_of_fit"] = exact_copy_count(synth, fit)
                row[f"{key}_copies_of_any_train_row"] = exact_copy_count(synth, frame)
                row[f"{key}_copies_of_fit_without_fnlwgt"] = exact_copy_count(synth, fit, NO_FNLWGT)
            # Sanity: the rebuilt test-sized draw is the one scored in results/holdout. The nearest-
            # neighbour distances depend on every value, so equal memorization numbers mean the same draw.
            saved = ROOT / "results" / "holdout" / f"{name}_seed{seed}.json"
            if saved.exists():
                ref = json.loads(saved.read_text())
                mem = memorization(d["test_synth"][FEATURES], fit[FEATURES], holdout[FEATURES], schema, seed)
                row["draw_rebuilt_exactly"] = bool(mem["dcr_fit_median"] == ref["dcr_fit_median"]
                                                   and mem["closer_to_fit_share"] == ref["closer_to_fit_share"])
            out[f"{name}_seed{seed}"] = row
            print(name, seed, {k: v for k, v in row.items() if "copies" in k}, flush=True)
            _write("exact_copies.json", out)
    return out


# ---------------------------------------------------------------- decoding ----

def _coverage_report(real_x, synth_x, schema):
    report = fidelity_report(real_x, synth_x, schema)
    missing = {}
    for column in CATEGORICAL:
        real_counts = real_x[column].astype(str).value_counts()
        synth_levels = set(synth_x[column].astype(str))
        gone = {level: int(n) for level, n in real_counts.items() if level not in synth_levels}
        if gone:
            missing[column] = gone
    return {
        "coverage": report["category_coverage_mean"], "rare_recall": report["rare_category_recall"],
        "tv": report["categorical_tv_mean"], "ks": report["numeric_ks_mean"],
        "coverage_by_column": {c: report["categorical"][c]["coverage"] for c in CATEGORICAL},
        "missing_categories": missing,
    }


def _decode(vae, z, mode, generator):
    num, logits = vae.decode(z)
    if mode == "argmax":
        cat = torch.stack([lg.argmax(1) for lg in logits], 1)
    else:
        cat = torch.stack([torch.multinomial(torch.softmax(lg, 1), 1, generator=generator)[:, 0]
                           for lg in logits], 1)
    return num, cat, logits


def run_decoding(seeds):
    frame = load_adult()
    schema = schema_from_frame(frame, NUMERIC, CATEGORICAL)
    out = {}
    for seed in seeds:
        fit, select, _ = split_frames(frame, seed)
        select_x, labels = select.drop(columns=["income"]), select["income"].to_numpy()
        saved = torch.load(MODEL_DIR / f"9_latent_rules_seed{seed}.pt", map_location="cpu", weights_only=False)
        sm = Sampler(saved)
        vae, enc = sm.vae.eval(), sm.encoder
        result = {"generation": {}, "reconstruction": {}, "softmax_mass": {}}

        # 1. Generated latents, decoded two ways, before and after the rule filter.
        for mode in ("argmax", "sample"):
            @torch.no_grad()
            def draw(lab, draw_seed, mode=mode):
                g = torch.Generator().manual_seed(int(draw_seed))
                y = torch.as_tensor(lab, dtype=torch.long)
                z = sm.model.sample(y, sm.dim, sm.config["sample_steps"], g)
                num, cat, _ = _decode(vae, z * sm.z_std + sm.z_mean, mode, g)
                return enc.inverse(num.numpy(), cat.numpy())
            for reject in (False, True):
                stats = {}
                synth = sm(labels, seed=7, stats=stats, reject=reject, draw=draw)[FEATURES]
                entry = _coverage_report(select_x, synth, schema)
                entry["rule_violation_share"] = float(rule_violations(synth)[0].mean())
                entry["rejected_share"] = stats.get("rejected_share")
                entry["selection_auc"] = selection_detection(select_x, synth, schema)["mean_auc"]
                result["generation"][f"{mode}{'+filter' if reject else ''}"] = entry
                print(seed, "gen", mode, reject, entry["coverage"], entry["rare_recall"], entry["selection_auc"],
                      flush=True)

        # 2. Average softmax mass the decoder gives each category on generated latents: a category
        # with real mass that argmax never emits points at the decoder, not the latent distribution.
        with torch.no_grad():
            g = torch.Generator().manual_seed(7)
            z = sm.model.sample(torch.as_tensor(labels, dtype=torch.long), sm.dim, sm.config["sample_steps"], g)
            _num, argmax_cat, logits = _decode(vae, z * sm.z_std + sm.z_mean, "argmax", g)
        # The encoder's categorical slots are the capital zero gates followed by CATEGORICAL.
        for j, column in enumerate(enc.cat_columns):
            if column not in CATEGORICAL:
                continue
            levels = list(enc.levels[column])
            probs = torch.softmax(logits[j], 1)
            mass = probs.mean(0).numpy()
            emitted = np.bincount(argmax_cat[:, j].numpy(), minlength=len(mass)) / len(argmax_cat)
            fit_share = fit[column].astype(str).value_counts(normalize=True)
            rows = []
            for k, level in enumerate(levels):
                rows.append({"level": str(level), "fit_share": float(fit_share.get(str(level), 0.0)),
                             "softmax_mass": float(mass[k]), "argmax_share": float(emitted[k]),
                             "max_prob": float(probs[:, k].max())})
            result["softmax_mass"][column] = rows

        # 3. Real rows through the VAE only (no diffusion): fit rows the VAE trained on, and unseen
        # selection rows. If these keep the rare categories, the loss comes from the latent model.
        for part, rows in (("fit", fit), ("selection", select)):
            x_num, x_cat = enc.transform(rows[FEATURES])
            x_num, x_cat = torch.tensor(x_num, dtype=torch.float32), torch.tensor(x_cat, dtype=torch.long)
            with torch.no_grad():
                mu, logvar = vae.encode(x_num, x_cat)
                for latent in ("mean", "posterior_sample"):
                    g = torch.Generator().manual_seed(seed)
                    z = mu if latent == "mean" else mu + torch.randn(mu.shape, generator=g) * (0.5 * logvar).exp()
                    for mode in ("argmax", "sample"):
                        num, cat, _ = _decode(vae, z.flatten(1), mode, g)
                        recon = enc.inverse(num.numpy(), cat.numpy())[FEATURES]
                        entry = _coverage_report(rows[FEATURES], recon, schema)
                        entry["category_accuracy"] = {c: float((cat[:, j] == x_cat[:, j]).float().mean())
                                                      for j, c in enumerate(enc.cat_columns)}
                        result["reconstruction"][f"{part}/{latent}/{mode}"] = entry
                        print(seed, "recon", part, latent, mode, entry["coverage"], entry["rare_recall"], flush=True)

        # 4. Reference: steps 7 and 8 on the same labels and draw seed.
        for step in ("7_ddpm", "8_ddpm_rules"):
            ref = Sampler(torch.load(MODEL_DIR / f"{step}_seed{seed}.pt", map_location="cpu", weights_only=False))
            synth = ref(labels, seed=7)[FEATURES]
            result["generation"][f"reference_{step}"] = _coverage_report(select_x, synth, schema)
        out[f"seed{seed}"] = result
        _write("decoding.json", out)
    return out


# ------------------------------------------------------------------ hybrid ----

def marginal_init(fit):
    """Each column drawn independently from the fit slice, given income: no joint structure at all."""
    by_label = {y: fit[fit["income"] == y][FEATURES].reset_index(drop=True) for y in (0, 1)}

    def init(labels, seed):
        rng = np.random.default_rng(int(seed) + 7)
        rows = pd.DataFrame(index=range(len(labels)), columns=FEATURES)
        for y, pool in by_label.items():
            idx = np.flatnonzero(labels == y)
            for column in FEATURES:
                rows.loc[idx, column] = pool[column].to_numpy()[rng.integers(len(pool), size=len(idx))]
        for column in NUMERIC:
            rows[column] = rows[column].astype(float)
        return rows
    return init


def gan_init(rung, seed):
    gan = load_generator(MODEL_DIR / f"{rung}_seed{seed}.pt", reject=False)
    return lambda labels, draw_seed: gan(labels, seed=draw_seed)[FEATURES]


def refine_draw(sampler, init, t0):
    """Start rows -> noised to t0 -> denoised by the step 8 model. Same as step 10 in evaluate_holdout."""
    model, encoder = sampler.model, sampler.encoder
    stride = model.T / sampler.config["sample_steps"]
    steps = max(1, round(t0 / stride))

    @torch.no_grad()
    def draw(labels, draw_seed):
        rows = init(labels, draw_seed)
        x_num, x_cat = encoder.transform(rows)
        g = torch.Generator().manual_seed(int(draw_seed) + 1)
        y = torch.as_tensor(labels, dtype=torch.long)
        t = torch.full((len(labels),), int(t0), dtype=torch.long)
        noisy = model.q_sample(torch.tensor(x_num), torch.tensor(x_cat), t, g)[:2]
        out_num, out_cat = model.sample(y, steps, g, start=noisy, t_start=t0)
        return encoder.inverse(out_num.numpy(), out_cat.numpy())
    return draw, steps


def pure_draw(sampler, steps):
    @torch.no_grad()
    def draw(labels, draw_seed):
        g = torch.Generator().manual_seed(int(draw_seed))
        num, cat = sampler.model.sample(torch.as_tensor(labels, dtype=torch.long), steps, g)
        return sampler.encoder.inverse(num.numpy(), cat.numpy())
    return draw


def run_hybrid(seeds, arms=None):
    frame, holdout = load_adult(), load_holdout()
    schema = schema_from_frame(frame, NUMERIC, CATEGORICAL)
    path = OUT_DIR / f"hybrid_seed{'-'.join(map(str, seeds))}.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for seed in seeds:
        fit, select, _ = split_frames(frame, seed)
        select_x = select.drop(columns=["income"])
        sampler = Sampler(torch.load(MODEL_DIR / f"8_ddpm_rules_seed{seed}.pt", map_location="cpu",
                                     weights_only=False))
        inits = {"init_gan5": lambda: gan_init("5_rules", seed), "init_gan4": lambda: gan_init("4_ctgan_norm", seed),
                 "init_marginals": lambda: marginal_init(fit)}
        plan = [(name, "init") for name in inits] + [(f"pure_{n}", "pure") for n in PURE_STEPS]
        for arm, kind in plan:
            key = f"{arm}_seed{seed}"
            if (arms and arm not in arms) or key in out:
                continue
            extra = {"arm": arm}
            if kind == "init":
                init = inits[arm]()
                scores = {}
                for t0 in REFINE_LEVELS_EXT:  # chosen on the selection slice only
                    draw, _ = refine_draw(sampler, init, t0)
                    rows = sampler(select["income"].to_numpy(), seed=42, reject=True, draw=draw)
                    scores[t0] = selection_detection(select_x, rows[FEATURES], schema)["mean_auc"]
                best = min(REFINE_LEVELS_EXT, key=lambda t0: (round(scores[t0], 4), t0))
                draw, steps = refine_draw(sampler, init, best)
                extra.update(refine_t0=best, refine_selection_auc=scores)
            else:
                steps = int(arm.split("_")[1])
                draw = pure_draw(sampler, steps)
            extra["denoiser_calls"] = steps
            generator = lambda labels, seed=0, stats=None, draw=draw: sampler(labels, seed, stats, reject=True,
                                                                              draw=draw)
            stats = {}
            dev, test_synth, utility_synth = draw_three(generator, seed, fit, select, holdout, stats)
            d = {"fit": fit, "dev_real": select_x, "dev": dev, "test_real": holdout, "test_synth": test_synth,
                 "utility_real": fit, "utility_synth": utility_synth, "stats": stats, "extra": extra}
            result = score_draws(arm, seed, d, holdout, schema)
            result["test_copies_of_fit"] = exact_copy_count(test_synth, fit)
            out[key] = result
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
            print(f"DONE {key}: t0 {extra.get('refine_t0')} calls {steps} RF {result['detection_auc_rf']:.4f} "
                  f"GB {result['detection_auc_gb']:.4f} util {result['utility_ratio']:.3f}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("check", choices=["copies", "decoding", "hybrid"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--arms", nargs="+", default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.time()
    {"copies": lambda: run_copies(args.seeds), "decoding": lambda: run_decoding(args.seeds),
     "hybrid": lambda: run_hybrid(args.seeds, args.arms)}[args.check]()
    print(f"finished in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()

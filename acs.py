"""Fresh evaluation on ACS California 2018 (ACSIncome task). Protocol: PROTOCOL.md.

  python acs.py train    --models 5_rules 7_ddpm 8_ddpm_rules 9_latent_rules --seeds 42
  python acs.py select   choices made on the validation part only (step 10's noise level,
                         step 9's decoding), written to results/acs/selection.json
  python acs.py evaluate the one-time test evaluation; refuses to run a second time

Training reads only the training and validation parts. The test part is read only by
`evaluate`, which writes a lock file first. Same configs as the Adult ladder (ladder.RUNGS,
diffusion.STEPS); nothing was tuned on ACS.
"""

import os

os.environ["TABGEN_DATASET"] = "acs"  # before importing ladder: switches columns and rules

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import acs_data  # noqa: E402
import diffusion  # noqa: E402
import ladder  # noqa: E402
from evaluator import schema_from_frame, selection_detection  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "acs"
MODEL_DIR = ladder.MODEL_DIR
SEEDS = ladder.SEEDS
FEATURES = ladder.NUMERIC + ladder.CATEGORICAL
REFINE_LEVELS = (25, 50, 100, 200, 400, 600, 800)
PURE_STEPS = (20, 40)


def environment():
    cpu = ""
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    import sklearn
    return {"cpu": cpu, "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "sklearn": sklearn.__version__, "threads": torch.get_num_threads(),
            "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()}


def load_parts(names):
    frame = acs_data.load_acs()
    parts = acs_data.load_splits(frame)
    frames = acs_data.part_frames(frame, parts, names)
    return frame, parts, frames


def schema_for(train):
    # Preprocessing reference = training part only (PROTOCOL.md, step 3).
    return schema_from_frame(train, ladder.NUMERIC, ladder.CATEGORICAL)


def membership(frame, parts):
    """Row positions and household ids per part: the membership labels for the inference study."""
    return {"rows": {k: v.tolist() for k, v in parts.items()},
            "households": {k: sorted(frame["SERIALNO"].iloc[v].unique().tolist()) for k, v in parts.items()},
            "sha256": acs_data.split_digest(parts), "members": "train"}


# ------------------------------------------------------------------- train ----

def run_train(models, seeds):
    frame, parts, (train, validation) = load_parts(("train", "validation"))
    schema = schema_for(train)
    split = membership(frame, parts)
    out_dir = RESULTS / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)
    for name in models:
        for seed in seeds:
            out = out_dir / f"{name}_seed{seed}.json"
            if out.exists():
                print("skip", out.name, flush=True)
                continue
            if name in ladder.RUNGS:
                result = ladder.train_rung(name, dict(ladder.RUNGS[name]), seed, train, validation, None, schema, log,
                                           model_dir=MODEL_DIR, split=split)
            else:
                result = diffusion.train_step(name, dict(diffusion.STEPS[name]), seed, train, validation, None,
                                              schema, log, split=split, model_dir=MODEL_DIR)
            result["environment"] = environment()
            out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
            print(f"DONE {name} seed {seed}: selected {result['selected']}", flush=True)


# ------------------------------------------------------------------ select ----

def _step9_sampler(seed, decode):
    saved = torch.load(MODEL_DIR / f"9_latent_rules_seed{seed}.pt", map_location="cpu", weights_only=False)
    saved["config"] = dict(saved["config"], decode=decode)
    return diffusion.Sampler(saved)


def _step8_sampler(seed):
    return diffusion.Sampler(torch.load(MODEL_DIR / f"8_ddpm_rules_seed{seed}.pt", map_location="cpu",
                                        weights_only=False))


def _inits(seed, train):
    from ablations import gan_init, marginal_init
    return {"10_gan_refine": lambda: gan_init("5_rules", seed), "10_marginals_refine": lambda: marginal_init(train)}


def run_select(seeds):
    """Every choice, on the validation part only (PROTOCOL.md, section 3)."""
    from ablations import refine_draw
    _frame, _parts, (train, validation) = load_parts(("train", "validation"))
    schema = schema_for(train)
    val_x, labels = validation[FEATURES], validation["income"].to_numpy()
    path = RESULTS / "selection.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for seed in seeds:
        entry = out.get(str(seed), {})
        if "step9_decode" not in entry:
            aucs = {}
            for decode in ("argmax", "sample"):
                rows = _step9_sampler(seed, decode)(labels, seed=42)
                aucs[decode] = selection_detection(val_x, rows[FEATURES], schema)["mean_auc"]
            # Ties keep argmax, the original decoder.
            entry["step9_decode"] = min(("argmax", "sample"), key=lambda d: (round(aucs[d], 4), d != "argmax"))
            entry["step9_validation_auc"] = aucs
        sampler = _step8_sampler(seed)
        for arm, make in _inits(seed, train).items():
            if f"{arm}_t0" in entry:
                continue
            init, scores = make(), {}
            for t0 in REFINE_LEVELS:
                draw, _ = refine_draw(sampler, init, t0)
                rows = sampler(labels, seed=42, reject=True, draw=draw)
                scores[t0] = selection_detection(val_x, rows[FEATURES], schema)["mean_auc"]
            entry[f"{arm}_t0"] = min(REFINE_LEVELS, key=lambda t0: (round(scores[t0], 4), t0))
            entry[f"{arm}_validation_auc"] = scores
        out[str(seed)] = entry
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
        print(seed, entry, flush=True)
    return out


# ---------------------------------------------------------------- evaluate ----

ARMS = ["real_reference", "5_rules", "7_ddpm", "7_ddpm+filter", "8_ddpm_rules", "9_latent_rules",
        "10_gan_refine", "10_marginals_refine", "pure_20", "pure_40"]


def _generator(arm, seed, train, choices):
    """Returns (generator(labels, seed, stats) -> frame, extra info)."""
    from ablations import pure_draw, refine_draw
    if arm == "5_rules":
        return ladder.load_generator(MODEL_DIR / f"5_rules_seed{seed}.pt"), {}
    if arm in ("7_ddpm", "7_ddpm+filter", "8_ddpm_rules"):
        step = arm.replace("+filter", "")
        reject = True if arm.endswith("+filter") else None
        saved = torch.load(MODEL_DIR / f"{step}_seed{seed}.pt", map_location="cpu", weights_only=False)
        sampler = diffusion.Sampler(saved)
        calls = saved["config"]["sample_steps"]
        return (lambda labels, seed=0, stats=None: sampler(labels, seed, stats, reject)), {"denoiser_calls": calls}
    if arm == "9_latent_rules":
        decode = choices["step9_decode"]
        sampler = _step9_sampler(seed, decode)
        return (lambda labels, seed=0, stats=None: sampler(labels, seed, stats)), {
            "decode": decode, "denoiser_calls": 2 * sampler.config["sample_steps"] - 1}
    sampler = _step8_sampler(seed)
    if arm.startswith("pure_"):
        draw, calls, extra = pure_draw(sampler, int(arm.split("_")[1])), int(arm.split("_")[1]), {}
    else:
        t0 = choices[f"{arm}_t0"]
        draw, calls = refine_draw(sampler, _inits(seed, train)[arm](), t0)
        extra = {"refine_t0": t0}
    extra["denoiser_calls"] = calls
    return (lambda labels, seed=0, stats=None: sampler(labels, seed, stats, reject=True, draw=draw)), extra


def run_evaluate(seeds):
    from ablations import exact_copy_count
    from evaluate_holdout import draw_three, score_draws
    test_dir = RESULTS / "test"
    lock = test_dir / "LOCK"
    if lock.exists():
        raise SystemExit(f"The test part was already read ({lock}). PROTOCOL.md allows one evaluation.")
    choices = json.loads((RESULTS / "selection.json").read_text())
    for seed in seeds:  # every checkpoint and choice must exist before the test part is opened
        assert str(seed) in choices, f"no validation choices for seed {seed}"
        for name in ("5_rules", "7_ddpm", "8_ddpm_rules", "9_latent_rules"):
            assert (MODEL_DIR / f"{name}_seed{seed}.pt").exists(), f"missing {name} seed {seed}"
    test_dir.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"opened": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "environment": environment()}, indent=2), encoding="utf-8")
    frame, parts, (train, validation, test, reference) = load_parts(("train", "validation", "test", "reference"))
    schema = schema_for(train)
    val_x = validation[FEATURES]
    unseen = {c: float((~test[c].isin(schema["categories"][c])).mean()) for c in ladder.CATEGORICAL}
    for seed in seeds:
        for arm in ARMS:
            stats, extra = {}, {}
            if arm == "real_reference":
                rng = np.random.default_rng(seed)
                pool = reference.iloc[rng.permutation(len(reference))].reset_index(drop=True)
                dev = pool.iloc[:len(validation)].reset_index(drop=True)
                test_synth = pool.iloc[len(validation):len(validation) + len(test)].reset_index(drop=True)
                utility_synth = test_synth.iloc[:len(train)]
            else:
                generator, extra = _generator(arm, seed, train, choices[str(seed)])
                dev, test_synth, utility_synth = draw_three(generator, seed, train, validation, test, stats)
            d = {"fit": train, "dev_real": val_x, "dev": dev, "test_real": test, "test_synth": test_synth,
                 "utility_real": train, "utility_synth": utility_synth, "stats": stats, "extra": extra}
            result = score_draws(arm, seed, d, test, schema)
            for key, rows in (("dev", dev), ("test_synth", test_synth), ("utility_synth", utility_synth)):
                result[f"{key}_copies_of_train"] = exact_copy_count(rows, train)
            result["generated_rows_checked"] = int(len(dev) + len(test_synth) + len(utility_synth))
            result["test_unseen_category_share"] = unseen
            result["environment"] = environment()
            (test_dir / f"{arm}_seed{seed}.json").write_text(json.dumps(result, indent=2, default=float),
                                                             encoding="utf-8")
            print(f"DONE {arm} seed {seed}: RF {result['detection_auc_rf']:.4f} GB {result['detection_auc_gb']:.4f} "
                  f"utility {result['utility_ratio']:.3f} coverage {result['coverage']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "select", "evaluate"])
    parser.add_argument("--models", nargs="+", default=["5_rules", "7_ddpm", "8_ddpm_rules", "9_latent_rules"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("LADDER_THREADS", "1")))
    started = time.time()
    if args.command == "train":
        run_train(args.models, args.seeds)
    elif args.command == "select":
        run_select(args.seeds)
    else:
        run_evaluate(args.seeds)
    print(f"finished in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    # Run through the importable modules so saved objects unpickle as ladder/diffusion classes.
    import acs

    acs.main()

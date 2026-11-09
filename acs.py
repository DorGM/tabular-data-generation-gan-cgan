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


def environment():
    cpu = ""
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    import sklearn
    return {"cpu": cpu, "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "sklearn": sklearn.__version__, "threads": torch.get_num_threads(),
            "git": _git_state()}


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


def file_sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


CHECKPOINTS = ("5_rules", "7_ddpm", "8_ddpm_rules", "9_latent_rules")


def checkpoint_hashes(seeds):
    return {f"{n}_seed{s}": file_sha256(MODEL_DIR / f"{n}_seed{s}.pt") for n in CHECKPOINTS for s in seeds}


def check_checkpoints(seeds, digest):
    """Every checkpoint exists, used the frozen config and the current split (no smoke-test leftovers)."""
    for name in CHECKPOINTS:
        frozen = ladder.RUNGS.get(name) or diffusion.STEPS[name]
        for seed in seeds:
            path = MODEL_DIR / f"{name}_seed{seed}.pt"
            if not path.exists():
                raise SystemExit(f"missing checkpoint {path}")
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if dict(saved["config"]) != dict(frozen):
                raise SystemExit(f"{path.name} was not trained with the frozen config")
            if saved["split"]["sha256"] != digest:
                raise SystemExit(f"{path.name} was trained on another split")


def run_select(seeds):
    """Every choice, on the validation part only (PROTOCOL.md, section 3)."""
    from ablations import refine_draw
    _frame, parts, (train, validation) = load_parts(("train", "validation"))
    check_checkpoints(seeds, acs_data.split_digest(parts))
    schema = schema_for(train)
    val_x, labels = validation[FEATURES], validation["income"].to_numpy()
    path = RESULTS / "selection.json"
    hashes = checkpoint_hashes(seeds)
    out = json.loads(path.read_text()) if path.exists() else {}
    if out.get("checkpoint_sha256", hashes) != hashes:
        out = {}  # checkpoints changed since the choices were made: redo them
    for seed in seeds:
        entry = out.get(str(seed), {})
        if "step9_decode" not in entry:
            aucs = {}
            for decode in ("argmax", "sample"):
                # Draw seed 43, not 42: seed 42's argmax draw already picked step 9's checkpoint.
                rows = _step9_sampler(seed, decode)(labels, seed=43)
                aucs[decode] = selection_detection(val_x, rows[FEATURES], schema)["mean_auc"]
            # Lower AUC at 4 decimals wins; a tie keeps argmax, the original decoder.
            entry["step9_decode"] = min(("argmax", "sample"), key=lambda d: (round(aucs[d], 4), d != "argmax"))
            entry["step9_validation_auc"] = aucs
        sampler = _step8_sampler(seed)
        for arm, make in _inits(seed, train).items():
            if f"{arm}_t0" in entry:
                continue
            init, scores = make(), {}
            for t0 in REFINE_LEVELS:  # same procedure as Adult's step 10 (draw seed 42)
                draw, _ = refine_draw(sampler, init, t0)
                rows = sampler(labels, seed=42, reject=True, draw=draw)
                scores[t0] = selection_detection(val_x, rows[FEATURES], schema)["mean_auc"]
            entry[f"{arm}_t0"] = min(REFINE_LEVELS, key=lambda t0: (round(scores[t0], 4), t0))
            entry[f"{arm}_validation_auc"] = scores
        out[str(seed)] = entry
        out["checkpoint_sha256"] = hashes
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
        print(seed, entry, flush=True)
    return out


# ---------------------------------------------------------------- evaluate ----

ARMS = ["real_reference", "5_rules", "7_ddpm", "7_ddpm+filter", "8_ddpm_rules", "9_latent_rules",
        "10_gan_refine", "10_marginals_refine", "pure_20", "pure_40", "pure_matched"]


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
        # pure_matched: the same number of denoiser calls as this seed's step 10 (t0 / 10).
        steps = max(1, round(choices["10_gan_refine_t0"] / 10)) if arm == "pure_matched" else int(arm.split("_")[1])
        draw, calls, extra = pure_draw(sampler, steps), steps, {}
    else:
        t0 = choices[f"{arm}_t0"]
        draw, calls = refine_draw(sampler, _inits(seed, train)[arm](), t0)
        extra = {"refine_t0": t0}
    extra["denoiser_calls"] = calls
    return (lambda labels, seed=0, stats=None: sampler(labels, seed, stats, reject=True, draw=draw)), extra


def _git_state():
    run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return {"head": run("rev-parse", "HEAD"), "dirty_tracked_files": run("status", "--porcelain", "-uno")}


def run_evaluate(seeds):
    """The one-time test evaluation. Everything that needs no test rows is checked before the lock."""
    from ablations import exact_copy_count
    from evaluate_holdout import draw_three, score_draws
    from evaluator import fidelity_report
    if tuple(seeds) != tuple(SEEDS):
        raise SystemExit(f"evaluate runs all seeds {SEEDS} at once")
    test_dir = RESULTS / "test"
    lock = test_dir / "LOCK"

    # --- checks before the lock: split, sizes, checkpoints, choices, every generator builds
    frame = acs_data.load_acs()
    parts = acs_data.load_splits(frame)
    digest = acs_data.split_digest(parts)
    n = {k: len(v) for k, v in parts.items()}
    if not (n["test"] >= n["train"] and n["reference"] >= n["validation"] + n["test"]):
        raise SystemExit(f"part sizes do not fit the real-reference construction: {n}")
    check_checkpoints(seeds, digest)
    selection_path = RESULTS / "selection.json"
    choices = json.loads(selection_path.read_text())
    hashes = checkpoint_hashes(seeds)
    if choices.get("checkpoint_sha256") != hashes:
        raise SystemExit("selection.json was made with other checkpoints; run `acs.py select` again")
    train, validation = acs_data.part_frames(frame, parts, ("train", "validation"))
    for seed in seeds:
        needed = ("step9_decode", "10_gan_refine_t0", "10_marginals_refine_t0")
        if any(k not in choices.get(str(seed), {}) for k in needed):
            raise SystemExit(f"validation choices missing for seed {seed}")
        for arm in ARMS[1:]:
            _generator(arm, seed, train, choices[str(seed)])
    state = {"seeds": list(seeds), "arms": ARMS, "split_sha256": digest, "checkpoint_sha256": hashes,
             "selection_sha256": file_sha256(selection_path), "git": _git_state()}

    # --- the lock. A rerun with the same state resumes (draws are seeded, so it is the same evaluation).
    if lock.exists():
        old = json.loads(lock.read_text())
        if {k: old.get(k) for k in state} != state:
            raise SystemExit(f"The test part was already read with a different state ({lock}). "
                             "PROTOCOL.md allows one evaluation.")
        print("resuming the one evaluation; finished arms are kept", flush=True)
    else:
        test_dir.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps({**state, "opened": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                    "environment": environment()}, indent=2), encoding="utf-8")

    test, reference = acs_data.part_frames(frame, parts, ("test", "reference"))
    schema = schema_for(train)
    val_x, test_x = validation[FEATURES], test[FEATURES]
    unseen = {c: float((~test[c].isin(schema["categories"][c])).mean()) for c in ladder.CATEGORICAL}
    for seed in seeds:
        for arm in ARMS:
            out = test_dir / f"{arm}_seed{seed}.json"
            if out.exists():
                continue
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
            report = fidelity_report(test_x, test_synth[FEATURES], schema)
            result["rare_category_recall"] = report["rare_category_recall"]
            result["rare_category_count"] = report["rare_category_count"]
            result["by_column"] = {c: {"coverage": report["categorical"][c]["coverage"],
                                       "rare_recall": report["categorical"][c]["rare_category_recall"],
                                       "tv": report["categorical"][c]["total_variation"]}
                                   for c in ladder.CATEGORICAL}
            result["correlation_mae"] = report["correlation_mae"]
            result["still_broken_after_filter"] = stats.get("still_broken")
            for key, rows in (("dev", dev), ("test_synth", test_synth), ("utility_synth", utility_synth)):
                result[f"{key}_copies_of_train"] = exact_copy_count(rows, train)
            result["generated_rows_checked"] = int(len(dev) + len(test_synth) + len(utility_synth))
            result["test_unseen_category_share"] = unseen
            result["environment"] = environment()
            out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
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

"""Aggregate results/holdout/*.json into SUMMARY.md and a figure. Does not train."""

import collections
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
HOLDOUT = ROOT / "results" / "holdout"
LABELS = {
    "real_holdout": "Real rows (unseen), reference",
    "0_baseline": "0 Baseline",
    "1_sampling": "1 + Sampling, integers, real zeros, base-rate labels",
    "2_st_onehot": "2 + Straight-through one-hots",
    "3_wgan_gp": "3 + WGAN-GP and PacGAN",
    "4_ctgan_norm": "4 + Mode-specific normalization, training-by-sampling",
    "4_ctgan_norm+reject": "4 + rule filter at sampling only",
    "5_rules": "5 + Domain-rule loss penalty and rule filter",
    "6_rules_snap": "6 + Capital values snapped to common real values",
    "7_ddpm": "7 Diffusion (TabDDPM style), no rules",
    "7_ddpm+reject": "7 + rule filter at sampling only",
    "8_ddpm_rules": "8 Diffusion + domain-rule loss and rule filter",
    "9_latent_rules": "9 Latent diffusion (TabSyn style) + rules",
    "10_gan_refine": "10 Step 5 GAN rows refined by step 8 diffusion",
    "sdv_gaussian_copula": "SDV Gaussian copula",
    "sdv_tvae": "SDV TVAE",
    "sdv_ctgan": "SDV CTGAN",
    "sdv_ctgan+reject": "SDV CTGAN + rule filter",
}
METRICS = [
    ("detection_auc_rf", "Detection AUC, forest ↓"),
    ("detection_auc_gb", "Detection AUC, boosting ↓"),
    ("utility_ratio", "Utility ratio ↑"),
    ("ks", "Numeric KS ↓"),
    ("tv", "Categorical TV ↓"),
    ("rule_violation_share", "Rule-breaking rows ↓"),
    ("closer_to_fit_share", "Closer to training rows (≈0.5 = no memorization)"),
    ("exact_copy_share", "Exact copies of training rows"),
    ("sample_seconds_per_1k", "Sampling seconds per 1,000 rows"),
]


def fmt(mean, sd):
    return "n/a" if np.isnan(mean) else f"{mean:.3f} ± {sd:.3f}"


def main():
    runs = collections.defaultdict(list)
    for path in sorted(HOLDOUT.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        runs[result["model"]].append(result)
    rows = []
    for key in LABELS:
        if key not in runs:
            continue
        entry = {"key": key, "label": LABELS[key], "n": len(runs[key])}
        for metric, _ in METRICS:
            values = np.array([np.nan if r.get(metric) is None else r[metric] for r in runs[key]], dtype=float)
            entry[metric] = (np.nanmean(values) if not np.isnan(values).all() else np.nan,
                             np.nanstd(values, ddof=1) if (~np.isnan(values)).sum() > 1 else 0.0)
        entry["snap_k"] = [r.get("snap_k") for r in runs[key] if r.get("snap_k")]
        rows.append(entry)

    lines = [
        "# Results on the official Adult test file (16,281 rows)",
        "",
        "Mean ± sample SD over split seeds 42, 68 and 124. Detectors are fit on selection rows vs. a synthetic",
        "draw and scored on holdout rows vs. a fresh draw. The reference row uses real rows the models never",
        "trained on, so it shows the best a generator could score. No model was trained on the test file and every",
        "choice was made on the selection slice, but the file was looked at many times while the ladder was built, so",
        "it is a reused benchmark. The exact-copy column compares 5,000 sampled synthetic",
        "rows with 10,000 sampled fit rows; ../ablations/SUMMARY.md checks every generated row against every fit row.",
        "",
        "| Model | " + " | ".join(name for _, name in METRICS) + " |",
        "| --- | " + " | ".join("---:" for _ in METRICS) + " |",
    ]
    for row in rows:
        lines.append(f"| {row['label']} | " + " | ".join(fmt(*row[m]) for m, _ in METRICS) + " |")
    refine = [r.get("refine_t0") for r in runs.get("10_gan_refine", []) if r.get("refine_t0")]
    if refine:
        lines += ["", f"Step 10 noise level t0 (of 1,000) chosen on the selection slice, per seed: {refine}."]
    snaps = [row for row in rows if row["snap_k"]]
    if snaps:
        lines += ["", f"Snap threshold k chosen on the selection slice, per seed: {snaps[0]['snap_k']}."]
    (HOLDOUT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    shown = [row for row in rows if not row["key"].endswith("+reject")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    colors = ["#55A868" if r["key"] == "real_holdout" else "#A0A0A0" if r["key"].startswith("sdv") else "#4C72B0"
              for r in shown]
    for axis, metric, title, xlim in [
        (axes[0], "detection_auc_rf", "Detection AUC, random forest (lower is better)", (0.45, 1.02)),
        (axes[1], "detection_auc_gb", "Detection AUC, gradient boosting (lower is better)", (0.45, 1.02)),
        (axes[2], "utility_ratio", "Utility ratio (higher is better)", (0.5, 1.05)),
    ]:
        axis.barh(range(len(shown)), [r[metric][0] for r in shown], xerr=[r[metric][1] for r in shown], color=colors)
        axis.set_yticks(range(len(shown)))
        axis.set_yticklabels([r["label"] for r in shown] if axis is axes[0] else [""] * len(shown), fontsize=8)
        axis.invert_yaxis()
        axis.set_xlim(*xlim)
        axis.set_title(title, fontsize=10)
        axis.grid(axis="x", alpha=0.3)
    fig.suptitle("Synthetic Adult data scored on the official Adult test file", fontsize=11)
    fig.tight_layout()
    fig.savefig(HOLDOUT / "holdout_results.png", dpi=150)
    print((HOLDOUT / "SUMMARY.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

"""Aggregate results/ladder/*.json into SUMMARY.md and a figure. Does not train."""

import collections
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
LADDER = ROOT / "results" / "ladder"
LABELS = {
    "0_baseline": "0 Baseline (published design)",
    "1_sampling": "1 + Sampling, integers, real zeros, base-rate labels",
    "2_st_onehot": "2 + Straight-through one-hots",
    "3_wgan_gp": "3 + WGAN-GP and PacGAN",
    "4_ctgan_norm": "4 + Mode-specific normalization, training-by-sampling",
    "5_rules": "5 + Domain-rule loss penalty and rule filter",
    "sdv_gaussian_copula": "SDV Gaussian copula",
    "sdv_tvae": "SDV TVAE",
    "sdv_ctgan": "SDV CTGAN",
}
METRICS = [
    ("final_detection_auc", "Detection AUC (lower is better)"),
    ("utility_ratio", "Utility ratio (higher is better)"),
    ("test_ks", "Numeric KS"),
    ("test_tv", "Categorical TV"),
    ("test_coverage", "Category coverage"),
]


def main():
    runs = collections.defaultdict(list)
    for path in sorted(LADDER.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        runs[result["rung"]].append(result)
    rows = []
    for key in LABELS:
        if key not in runs:
            continue
        group = runs[key]
        entry = {"key": key, "label": LABELS[key], "n": len(group)}
        for metric, _name in METRICS:
            values = np.array([run[metric] for run in group], dtype=float)
            entry[metric] = (values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0)
        entry["violations"] = sum(int(run["test_violations"]) for run in group)
        rows.append(entry)

    lines = [
        "# Improvement ladder: results on the held-out test split",
        "",
        "Mean ± sample SD over split seeds 42, 68 and 124. Each rung adds one change to the rung above it.",
        "Checkpoints were chosen on the selection slice; the test split was scored once per run.",
        "",
        "| Model | Seeds | Detection AUC ↓ | Utility ratio ↑ | Numeric KS ↓ | Categorical TV ↓ | Coverage ↑ | Hard-validity violations |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        cells = [f"{row[m][0]:.3f} ± {row[m][1]:.3f}" for m, _ in METRICS]
        lines.append(f"| {row['label']} | {row['n']} | " + " | ".join(cells) + f" | {row['violations']} |")
    (LADDER / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    names = [row["label"].split(" (")[0] for row in rows]
    colors = ["#4C72B0" if not row["key"].startswith("sdv") else "#A0A0A0" for row in rows]
    for axis, (metric, title) in zip(axes, METRICS[:2]):
        means = [row[metric][0] for row in rows]
        errors = [row[metric][1] for row in rows]
        axis.barh(range(len(rows)), means, xerr=errors, color=colors)
        axis.set_yticks(range(len(rows)))
        axis.set_yticklabels(names if axis is axes[0] else [""] * len(rows), fontsize=8)
        axis.invert_yaxis()
        axis.set_title(title, fontsize=10)
        axis.set_xlim(0.5, 1.02)
        axis.grid(axis="x", alpha=0.3)
        if metric == "final_detection_auc":
            axis.axvline(0.5, color="black", linewidth=0.8)
    fig.suptitle("Synthetic Adult data: each fix vs. off-the-shelf generators (test split)", fontsize=11)
    fig.tight_layout()
    fig.savefig(LADDER / "ladder_results.png", dpi=150)
    print((LADDER / "SUMMARY.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

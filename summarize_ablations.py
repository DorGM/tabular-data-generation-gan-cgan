"""Builds results/ablations/SUMMARY.md from the JSON files written by ablations.py."""

import json
from pathlib import Path

import numpy as np

from ladder import SEEDS

ROOT = Path(__file__).resolve().parent
ABL = ROOT / "results" / "ablations"
HOLDOUT = ROOT / "results" / "holdout"


def ms(values, digits=3):
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}" if len(values) > 1 else f"{values[0]:.{digits}f}"


def copies_section():
    path = ABL / "exact_copies.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    order = list(dict.fromkeys(k.rsplit("_seed", 1)[0] for k in data))
    ref = json.loads((ABL / "real_copy_rate.json").read_text())
    lines = ["## Copies of training rows over every generated row", "",
             "Every row of the three evaluation draws (selection-sized, test-sized and fit-sized), for all three seeds, "
             "compared with every fit-slice row. \"All 13 columns\" is an exact copy. Because `fnlwgt` is close to "
             "unique per person, the 12 other columns are also checked (a quasi-identifier match). Step 0 does not "
             "round numbers, so it cannot match exactly by construction.", "",
             f"Scale from real data: of {ref['rows']:,} real selection-slice rows that no model trained on, "
             f"{ref['matches']} ({100 * ref['matches'] / ref['rows']:.2f}%) equal a fit row in all 13 columns and "
             f"{ref['matches_without_fnlwgt']:,} ({100 * ref['matches_without_fnlwgt'] / ref['rows']:.1f}%) "
             "in the 12 columns without `fnlwgt`.", "",
             "| Model | Rows checked | Copies, all 13 columns | Matches without fnlwgt | Draw rebuilt exactly |",
             "| --- | ---: | ---: | ---: | :---: |"]
    keys = ("dev", "test_synth", "utility_synth")
    for m in order:
        rows = [data[f"{m}_seed{s}"] for s in SEEDS if f"{m}_seed{s}" in data]
        checked = sum(r[f"{k}_rows"] for r in rows for k in keys)
        exact = sum(r[f"{k}_copies_of_fit"] for r in rows for k in keys)
        quasi = sum(r[f"{k}_copies_of_fit_without_fnlwgt"] for r in rows for k in keys)
        rebuilt = [r.get("draw_rebuilt_exactly") for r in rows]
        flag = "yes" if all(rebuilt) else ("no (unseeded SDV sampler)" if m.startswith("sdv") else "no")
        partial = "" if len(rows) == len(SEEDS) else f" (only {len(rows)} seeds so far)"
        lines.append(f"| {m}{partial} | {checked:,} | {exact} ({100 * exact / checked:.3f}%) | "
                     f"{quasi:,} ({100 * quasi / checked:.1f}%) | {flag} |")
    return lines + [""]


def decoding_section():
    path = ABL / "decoding.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    seeds = list(data)
    lines = ["## Step 9: where rare categories are lost", "",
             "Coverage = share of categories present in the real rows that appear at least once; rare recall = the same "
             "for categories under 1% frequency, measured against the real rows of the same slice.", "",
             "| Rows | Decoding | Coverage | Rare recall | Categorical TV | Selection AUC |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    gen = [("Generated latents", "argmax (current)", "argmax+filter"), ("Generated latents", "softmax sample", "sample+filter")]
    for label, name, key in gen:
        e = [data[s]["generation"][key] for s in seeds]
        lines.append(f"| {label}, with filter | {name} | {ms([x['coverage'] for x in e])} | "
                     f"{ms([x['rare_recall'] for x in e])} | {ms([x['tv'] for x in e])} | "
                     f"{ms([x['selection_auc'] for x in e], 4)} |")
    for part in ("fit", "selection"):
        for mode, name in (("argmax", "argmax"), ("sample", "softmax sample")):
            e = [data[s]["reconstruction"][f"{part}/mean/{mode}"] for s in seeds]
            lines.append(f"| Real {part} rows through the VAE only | {name} | {ms([x['coverage'] for x in e])} | "
                         f"{ms([x['rare_recall'] for x in e])} | {ms([x['tv'] for x in e])} | - |")
    for step in ("7_ddpm", "8_ddpm_rules"):
        e = [data[s]["generation"][f"reference_{step}"] for s in seeds]
        lines.append(f"| Step {step[0]} (reference) | - | {ms([x['coverage'] for x in e])} | "
                     f"{ms([x['rare_recall'] for x in e])} | {ms([x['tv'] for x in e])} | - |")
    # Categories argmax never emits, with the decoder's average probability for them.
    rows = []
    for c, entries in data[seeds[0]]["softmax_mass"].items():
        for x in entries:
            if x["fit_share"] > 0 and x["argmax_share"] == 0:
                rows.append((c, x))
    top = max(x["max_prob"] for _c, x in rows)
    lines += ["", f"In {seeds[0]}, argmax never emits {len(rows)} categories that occur in the fit slice: none of them "
                  f"is the most likely category in any generated row (highest probability {top:.2f}). The 12 most "
                  "frequent are shown:", "",
              "| Column | Category | Fit share | Mean softmax probability | Highest probability in any row |",
              "| --- | --- | ---: | ---: | ---: |"]
    for c, x in sorted(rows, key=lambda r: -r[1]["fit_share"])[:12]:
        lines.append(f"| {c} | {x['level']} | {x['fit_share']:.4f} | {x['softmax_mass']:.4f} | {x['max_prob']:.3f} |")
    return lines + [""]


def hybrid_section():
    data = {}
    for path in sorted(ABL.glob("hybrid_seed*.json")):
        data.update(json.loads(path.read_text()))
    if not data:
        return []
    ref = {}
    for name in ("8_ddpm_rules", "10_gan_refine"):
        ref[name] = [json.loads((HOLDOUT / f"{name}_seed{s}.json").read_text()) for s in SEEDS]
    arms = [("10_gan_refine", "Step 10 as published: GAN 5 start, t0 grid up to 400"),
            ("init_gan5", "GAN step 5 start, t0 grid up to 800"),
            ("init_gan4", "GAN step 4 start (no rules), t0 grid up to 800"),
            ("init_marginals", "Independent-marginals start, t0 grid up to 800"),
            ("pure_20", "Pure diffusion, 20 steps"), ("pure_40", "Pure diffusion, 40 steps"),
            ("pure_60", "Pure diffusion, 60 steps"), ("pure_80", "Pure diffusion, 80 steps"),
            ("8_ddpm_rules", "Pure diffusion, 100 steps (step 8)")]
    lines = ["## Step 10: does the GAN start matter?", "",
             "All rows use the same step 8 denoiser and rule filter. For the start variants, t0 is chosen on the "
             "selection slice per seed. Denoiser calls = t0 / 10. Test-file numbers (a reused benchmark).", "",
             "| Variant | t0 per seed | Denoiser calls | Forest AUC | Boosting AUC | Utility | Coverage |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for arm, label in arms:
        rows = ref.get(arm) or [data.get(f"{arm}_seed{s}") for s in SEEDS]
        rows = [r for r in rows if r]
        if not rows:
            continue
        t0 = ", ".join(str(r.get("refine_t0", "-")) for r in rows)
        calls = ", ".join(str(r.get("denoiser_calls") or (r["refine_t0"] // 10 if r.get("refine_t0") else 100))
                          for r in rows)
        lines.append(f"| {label} | {t0} | {calls} | {ms([r['detection_auc_rf'] for r in rows])} | "
                     f"{ms([r['detection_auc_gb'] for r in rows])} | {ms([r['utility_ratio'] for r in rows])} | "
                     f"{ms([r['coverage'] for r in rows])} |")
    return lines + [""]


def main():
    lines = ["# Follow-up checks on the Adult benchmark", "",
             "Built by `summarize_ablations.py` from `ablations.py` runs on the saved checkpoints (no retraining). "
             "Mean ± SD over the three split seeds.", ""]
    lines += copies_section() + decoding_section() + hybrid_section()
    (ABL / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

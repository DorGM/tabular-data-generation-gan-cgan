"""Builds results/acs/SUMMARY.md from the one-time test evaluation (acs.py evaluate).

The hypothesis verdicts apply the decision rules written in PROTOCOL.md section 7 before the
test part was read: boosting AUC is primary, and a difference counts only if the 3-seed means
differ by at least 0.01 in the stated direction.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ACS = ROOT / "results" / "acs"
SEEDS = (42, 68, 124)
MARGIN = 0.01
LABELS = {
    "real_reference": "Real reference rows (best reachable)",
    "5_rules": "5 GAN + rules",
    "7_ddpm": "7 Diffusion, no rules, no filter",
    "7_ddpm+filter": "7 Diffusion + rule filter",
    "8_ddpm_rules": "8 Diffusion + rule penalty + filter",
    "9_latent_rules": "9 Latent diffusion + rules",
    "10_gan_refine": "10 GAN rows refined by step 8",
    "10_marginals_refine": "Control: independent-marginals start refined by step 8",
    "pure_matched": "Control: step 8 from noise, same calls as step 10",
    "pure_20": "Control: step 8 from noise, 20 calls",
    "pure_40": "Control: step 8 from noise, 40 calls",
}


def load():
    runs = {}
    for arm in LABELS:
        rows = [json.loads(p.read_text()) for s in SEEDS if (p := ACS / "test" / f"{arm}_seed{s}.json").exists()]
        if rows:
            runs[arm] = rows
    return runs


def mean(runs, arm, key):
    values = [r[key] for r in runs[arm] if r.get(key) is not None]
    return float(np.mean(values)) if values else None


def fmt(runs, arm, key, digits=3, scale=1.0, suffix=""):
    values = [r[key] * scale for r in runs[arm] if r.get(key) is not None]
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}{suffix}"
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}{suffix}"


def column(runs, arm, column, key="coverage"):
    values = [r["by_column"][column][key] for r in runs[arm] if r["by_column"][column][key] is not None]
    return f"{np.mean(values):.3f}" if values else "-"


def copies(runs, arm):
    total = sum(r[f"{k}_copies_of_train"] for r in runs[arm] for k in ("dev", "test_synth", "utility_synth"))
    checked = sum(r["generated_rows_checked"] for r in runs[arm])
    return f"{total} / {checked:,}"


def verdict(ok, close):
    return "supported" if ok else ("no measurable difference" if close else "not supported")


def hypotheses(runs):
    gb = lambda arm: mean(runs, arm, "detection_auc_gb")
    rf = lambda arm: mean(runs, arm, "detection_auc_rf")
    cov = lambda arm: mean(runs, arm, "coverage")
    lines = ["## Pre-registered hypotheses (PROTOCOL.md section 7)", "",
             f"Boosting AUC is primary. A difference counts only if the 3-seed means differ by at least {MARGIN}.", ""]
    d71, d81 = gb("5_rules") - gb("7_ddpm+filter"), gb("5_rules") - gb("8_ddpm_rules")
    lines.append(f"1. **Diffusion beats the GAN.** Boosting AUC: 5 = {gb('5_rules'):.3f}, 7 + filter = "
                 f"{gb('7_ddpm+filter'):.3f}, 8 = {gb('8_ddpm_rules'):.3f}. Verdict: "
                 f"**{verdict(d71 >= MARGIN and d81 >= MARGIN, max(abs(d71), abs(d81)) < MARGIN)}**.")
    redraw = mean(runs, "8_ddpm_rules", "rejected_share")
    gap = gb("8_ddpm_rules") - gb("7_ddpm+filter")
    redraw7 = mean(runs, "7_ddpm+filter", "rejected_share")
    lines.append(f"2. **The rule penalty removes most rule-breaking.** Filter redraws: step 7 {100 * redraw7:.2f}%, "
                 f"step 8 {100 * redraw:.3f}% (target under 0.1%). Boosting AUC of 8 minus 7 + filter: {gap:+.3f} "
                 f"(limit +{MARGIN}). Verdict: **{'supported' if redraw < 0.001 and gap <= MARGIN else 'not supported'}**.")
    c9, c78 = cov("9_latent_rules"), min(cov("7_ddpm+filter"), cov("8_ddpm_rules"))
    decode = sorted({r.get("decode") for r in runs["9_latent_rules"]})
    lines.append(f"3. **Latent diffusion keeps fewer categories.** Column-mean coverage: 9 = {c9:.3f} (decoding "
                 f"chosen on validation: {', '.join(decode)}), lower of 7 + filter and 8 = {c78:.3f}. Verdict: "
                 f"**{verdict(c78 - c9 >= MARGIN, abs(c78 - c9) < MARGIN)}**.")
    dp, dm = gb("pure_matched") - gb("10_gan_refine"), gb("10_marginals_refine") - gb("10_gan_refine")
    rp, rm = rf("pure_matched") - rf("10_gan_refine"), rf("10_marginals_refine") - rf("10_gan_refine")
    lines.append(f"4. **The GAN start helps.** Boosting AUC of step 10 = {gb('10_gan_refine'):.3f}; pure diffusion at "
                 f"the same calls minus step 10: {dp:+.3f}; marginals start minus step 10: {dm:+.3f} (forest: {rp:+.3f} and "
                 f"{rm:+.3f}). Verdict on quality: **{verdict(dp >= MARGIN and dm >= MARGIN, max(abs(dp), abs(dm)) < MARGIN)}**.")
    return lines + [""]


def main():
    runs = load()
    selection = json.loads((ACS / "selection.json").read_text())
    lock = json.loads((ACS / "test" / "LOCK").read_text())
    lines = ["# Fresh evaluation on ACS California 2018 (ACSIncome)", "",
             f"One-time evaluation on the reserved test part (30,000 people, households kept together), opened "
             f"{lock['opened']}. Protocol: `PROTOCOL.md`. Every model was fit on the training part (30,000 people) and "
             "every choice was made on the validation part (8,000). Mean ± SD over training seeds 42, 68 and 124 on one "
             "fixed split.", "",
             "| Model | Forest AUC ↓ | Boosting AUC ↓ | Utility ↑ | KS ↓ | TV ↓ | Coverage ↑ | Rare recall ↑ | "
             "Occupation coverage | Birthplace coverage | Rule-breaking rows | Filter redraws | Closer to training "
             "(0.5 = none) | Exact copies / rows | Denoiser calls | Seconds per 1k rows |",
             "| --- |" + " ---: |" * 15]
    for arm in LABELS:
        if arm not in runs:
            continue
        calls = ", ".join(str(r.get("denoiser_calls", "-")) for r in runs[arm])
        lines.append(
            f"| {LABELS[arm]} | {fmt(runs, arm, 'detection_auc_rf')} | {fmt(runs, arm, 'detection_auc_gb')} | "
            f"{fmt(runs, arm, 'utility_ratio')} | {fmt(runs, arm, 'ks')} | {fmt(runs, arm, 'tv')} | "
            f"{fmt(runs, arm, 'coverage')} | {fmt(runs, arm, 'rare_category_recall')} | {column(runs, arm, 'OCCP')} | "
            f"{column(runs, arm, 'POBP')} | {fmt(runs, arm, 'rule_violation_share', 2, 100, '%')} | "
            f"{fmt(runs, arm, 'rejected_share', 3, 100, '%')} | {fmt(runs, arm, 'closer_to_fit_share')} | "
            f"{copies(runs, arm)} | {calls} | {fmt(runs, arm, 'sample_seconds_per_1k', 2)} |")
    lines += ["", "Validation choices per seed (`results/acs/selection.json`):", ""]
    for seed in SEEDS:
        c = selection[str(seed)]
        lines.append(f"- Seed {seed}: step 9 decoding **{c['step9_decode']}**, step 10 t0 **{c['10_gan_refine_t0']}**, "
                     f"marginals-start t0 **{c['10_marginals_refine_t0']}**.")
    lines += [""] + hypotheses(runs)
    env = runs["5_rules"][0]["environment"]
    lines += [f"Environment: {env['cpu']}, torch {env['torch']}, numpy {env['numpy']}, scikit-learn {env['sklearn']}, "
              f"{env['threads']} thread per run.", ""]
    (ACS / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

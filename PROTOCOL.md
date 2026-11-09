# Fresh evaluation protocol: ACS California 2018

This protocol was written and committed before any model was evaluated on the test part. The
Adult test file has been looked at many times while the ladder was built, so it is a reused
benchmark. This round checks the main Adult comparisons on data that played no part in any
design decision.

## 1. Population and task (fixed in advance)

- **Source:** 2018 ACS 1-year Public Use Microdata Sample, California person file (`psam_p06.csv`
  from `csv_pca.zip`, www2.census.gov).
- **Task:** the folktables ACSIncome task, which mimics the Adult extraction.
  - Filter: age over 16, total income over $100, usual hours worked over 0, person weight at least 1. This leaves 195,665 people in 111,426 households.
  - Label: total person income (`PINCP`) over $50,000.
  - Features:
    - Numeric (integers): age (`AGEP`) and usual hours per week (`WKHP`).
    - Categorical codes: class of worker (`COW`), education (`SCHL`), marital status (`MAR`), occupation (`OCCP`, 529 codes), place of birth (`POBP`, 219 codes), relationship to the reference person (`RELP`), sex (`SEX`) and race (`RAC1P`).
- Survey weights are not used. The rows are treated as a sample, as with Adult.

## 2. Splits (fixed, by household)

- Households (`SERIALNO`) are shuffled once with `numpy.random.default_rng(2018)`. Parts are filled in that order, with whole households only, until each reaches its size:

| Part | People | Use |
| --- | ---: | --- |
| train | 30,000 | Fits every encoder and generator. These rows are the **members** for the inference study. |
| validation | 8,000 | Checkpoint selection, step 10's noise level, step 9's decoding mode. Also the real rows the detectors are fit on. |
| test | 30,000 | Read once, by `acs.py evaluate`. |
| reference | 38,003 | Real rows that stand in for a perfect generator, giving the best reachable scores. |
| reserve | 89,662 | Not read by any script in this round; kept for a later fresh check. |

- The row positions and household ids of every part are saved with a SHA-256 digest. Every script rebuilds the split and refuses to run if the digest differs. Digest: `91322ba81fdc0a3e5cf730c787e87cba745cc8875a5261376d597c48f7cb4282`.
- No household appears in two parts. A test checks this.

## 3. What is fit where

- **Training part only:** encoders (quantile and Gaussian-mixture transforms, category vocabularies), the evaluator's schema (numeric domain, category lists), all generators, and the domain rules.
- **Validation part only:** every choice.
  - Checkpoints use the same rule as on Adult: zero hard-validity violations first, then the lowest detection AUC, then the earlier checkpoint.
  - Step 10's noise level t0, from 25, 50, 100, 200, 400, 600 and 800 out of 1,000.
  - Step 9's categorical decoding: argmax or sampling from the decoder's softmax, whichever has the lower validation detection AUC. This choice was added after the Adult check on rare categories (`results/ablations/decoding.json`).
- **Hyperparameters:** taken from the Adult configs (`ladder.RUNGS`, `diffusion.STEPS`), with nothing tuned on ACS. The same number of epochs or steps is used, and the same three training seeds (42, 68, 124) on the one fixed split.

## 4. Domain rules (derived from the training part only)

Each rule follows from what the fields mean. A rule is kept only if it has at most 1 exception per 10,000 training rows:

| Rule | Exceptions in train |
| --- | ---: |
| A spouse of the reference person (`RELP` 1) is married (`MAR` 1). | 0 |
| An unmarried partner (`RELP` 13) is not married. | 0 |
| Armed-forces occupations (`OCCP` 9800 to 9830) work for the federal government (`COW` 5). | 1 |

Adult's husband/wife-sex rule does not carry over, because the 2018 ACS spouse code does not depend on sex. Adult's missing-occupation rule does not apply either, because ACSIncome keeps only people who work.

## 5. Models (retrained on ACS)

| Name | What it is |
| --- | --- |
| `5_rules` | GAN with mode-specific normalization, training-by-sampling, rule penalty and rule filter |
| `7_ddpm+filter` | TabDDPM-style diffusion, no rule loss, rule filter at sampling |
| `7_ddpm` | the same model without the filter, reported for the share of rule-breaking rows |
| `8_ddpm_rules` | diffusion with the rule penalty and the filter |
| `9_latent_rules` | TabSyn-style latent diffusion with rules; decoding chosen on validation |
| `10_gan_refine` | step 5 rows noised to t0 and denoised by step 8, with the filter |

Controls that need no extra training:

- `pure_20` and `pure_40`: step 8 sampled from noise with 20 and 40 denoiser calls. Step 10 uses 2 to 80 calls, depending on t0.
- `10_marginals_refine`: the same refinement as step 10, but starting from columns drawn independently from the training part, given the label. Its t0 is chosen on validation.
- `real_reference`: real reference rows in place of synthetic rows.

## 6. Metrics (computed once on the test part)

These are the same as on Adult:

- detection AUC with a random forest and with gradient boosting, each fit on validation rows vs. one synthetic draw and scored on test rows vs. a fresh draw;
- train-on-synthetic, test-on-real utility on the test part;
- KS, categorical TV, category coverage and rare-category recall;
- hard validity and rule violations;
- the share of rows the filter redraws;
- sampling seconds per 1,000 rows and denoiser calls;
- exact copies of training rows over every generated row;
- the memorization share: is each synthetic row closer to the training rows than to the test rows?

Mean ± SD over the three training seeds. Seed SD understates run-to-run variation (the same seed differs across CPUs), so differences under about 0.01 AUC are not interpreted. The CPU model and library versions are saved with every result.

## 7. Hypotheses stated before the test is read

These come from the Adult results:

1. Diffusion with the filter (7 + filter, 8) has a lower boosting AUC than the GAN with rules (5).
2. The rule penalty (8 vs. 7) brings the filter's redraw share close to zero without hurting detection AUC.
3. Latent diffusion (9) has lower category coverage than 7 and 8.
4. Step 10 is no better than pure diffusion at the same number of denoiser calls, and no better than refinement from independent marginals. If it is better than both, the GAN start helps.

## 8. One-time evaluation

- `acs.py evaluate` writes a lock file before reading the test part and refuses to run if one exists.
- Every result is reported, including ones that go against the hypotheses.
- If a bug is found after the test was read, the fix and both sets of numbers are reported, and the rerun is marked as such.

## 9. Kept for the membership-inference study

Every checkpoint keeps the same contents as on Adult:

- selected and final weights, optimizer states, four snapshots and loss curves;
- the GAN critic and the encoders;
- the split (row positions and household ids).

Training rows are the members. The test and reference parts are non-members from the same population, held out by household.

# Synthetic Tabular Data with GANs and Diffusion on UCI Adult

Conditional GANs and diffusion models that generate realistic synthetic census rows, evaluated on the 16,281 rows of the official Adult test file, which no model trained on. The best models reach a detection AUC of about **0.70 to 0.73**, where 0.5 means indistinguishable and 1.0 means always caught. That holds for both a random-forest and a gradient-boosting detector. These models keep **96 to 98%** of real-data utility, every row satisfies the domain rules, and at most one of the roughly 127,000 rows generated per model during evaluation is an exact copy of a training row. Under the same protocol, SDV's CTGAN scores 0.935 and TVAE 0.937.

## Why I built this

Synthetic data is used to share sensitive tables (census, medical, financial) and to augment small datasets. Before anyone uses it, a data scientist has to answer two questions: is it realistic, and is it useful? Plausible-looking samples or a falling loss curve answer neither.

This project builds tabular generators and an evaluation that answers both:

- **Realism** is measured as detectability. Two different classifiers try to tell real rows from synthetic rows.
- **Usefulness** is measured by training a model on synthetic data and testing it on real data.
- **Safety** is checked by testing whether synthetic rows sit closer to the training rows than to unseen real rows.

Each step adds one change, or a named bundle of related changes, and is measured on its own. Steps 1, 3, 4 and 5 bundle several changes each, so their parts are not measured separately. Steps 0 to 5 improve a conditional GAN. Steps 7 to 10 test diffusion models and hybrids under the same protocol, with the same domain rules.

This tabular work complements my computer vision projects. I'm planning a follow-up that combines conditional generation with my face-verification pipeline.

## Results

All numbers are on the official Adult test file (16,281 rows). No model was trained on it, and every checkpoint, threshold and noise level was chosen on the selection slice. The file has been looked at many times while the ladder was built, though, so it is a reused benchmark rather than a fresh one. Values are the mean ± SD over three split seeds. The same seed can differ by about 0.01 AUC on a different CPU, so smaller differences should not be read as real. The first row uses real rows the models never trained on, which is the best score any generator could reach.

| Model | Detection AUC, forest ↓ | Detection AUC, boosting ↓ | Utility ratio ↑ | Numeric KS ↓ | Rule-breaking rows | Closer to training rows (0.5 = no memorization) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Real unseen rows (reference) | 0.499 ± 0.005 | 0.504 ± 0.004 | 1.000 | 0.015 | 0% | 0.501 |
| **GAN ladder** | | | | | | |
| 0 Baseline cGAN | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.617 | 0.737 | 41.7% | 0.607 (unstable) |
| 1 + Sampling, integers, real zeros, base-rate labels | 0.974 ± 0.010 | 0.981 ± 0.008 | 0.842 | 0.154 | 0.2% | 0.508 |
| 2 + Straight-through one-hots | 0.947 ± 0.006 | 0.965 ± 0.004 | 0.910 | 0.140 | 0.2% | 0.501 |
| 3 + WGAN-GP and PacGAN | 0.936 ± 0.005 | 0.960 ± 0.003 | 0.952 | 0.109 | 7.2% | 0.492 |
| 4 + Mode-specific normalization, training-by-sampling | 0.781 ± 0.005 | 0.826 ± 0.007 | 0.945 | 0.029 | 7.1% | 0.498 |
| 5 + Domain-rule loss and rule filter | 0.749 ± 0.004 | 0.808 ± 0.006 | 0.960 | 0.041 | 0% | 0.501 |
| 6 = 5 + capital values snapped to common real values | 0.747 ± 0.004 | 0.807 ± 0.003 | 0.961 | 0.040 | 0% | 0.500 |
| **Diffusion and hybrids** | | | | | | |
| 7 Diffusion (TabDDPM style), no rules | 0.764 ± 0.015 | 0.773 ± 0.012 | 0.982 | 0.015 | 23.0% | 0.502 |
| 7 + rule filter at sampling only | 0.711 ± 0.010 | **0.716 ± 0.013** | 0.981 | 0.020 | 0% | 0.509 |
| 8 Diffusion + domain-rule loss and rule filter | 0.716 ± 0.012 | 0.718 ± 0.013 | 0.963 | 0.016 | 0% | 0.509 |
| 9 Latent diffusion (TabSyn style) + rules | **0.700 ± 0.028** | 0.732 ± 0.028 | **0.983** | 0.030 | 0% | 0.500 |
| 10 Step 5 GAN rows refined by step 8 diffusion | 0.705 ± 0.008 | 0.717 ± 0.009 | 0.980 | 0.026 | 0% | 0.495 |
| **SDV references** | | | | | | |
| SDV Gaussian copula | 0.998 ± 0.001 | 0.999 ± 0.001 | 0.868 | 0.326 | 33.2% | 0.484 |
| SDV TVAE | 0.937 ± 0.012 | 0.950 ± 0.009 | 0.964 | 0.161 | 3.0% | 0.491 |
| SDV CTGAN | 0.935 ± 0.019 | 0.953 ± 0.013 | 0.967 | 0.168 | 8.1% | 0.519 |

![Detection AUC and utility on the official test file](results/holdout/holdout_results.png)

Full table, including categorical TV, exact copies, sampling time and the extra control runs: `results/holdout/SUMMARY.md`.

**What the results show**

- **Targets fixed before the work.**
  - For the GAN: detection AUC at most 0.75, utility at least 0.95, and zero invalid rows. Step 5 meets all three with the random-forest detector (0.749).
  - For the diffusion steps: a boosting AUC below step 5's 0.80, utility at least 0.95, no exact copies and a memorization share near 0.5. Steps 8, 9 and 10 meet all four. Step 7 + filter has one exact copy in 127,000 generated rows. That is far below the rate for unseen real rows, 0.11% of which match a training row.
- **Diffusion closes most of the gap the boosting detector found.** It falls from 0.808 for the best GAN to about 0.72. The two detectors now agree, which suggests the GAN's remaining flaws were patterns that boosting finds and a forest misses.
- **There is no clear winner among the diffusion models.** Steps 7 + filter, 8, 9 and 10 are within about two seed SDs of each other. Each has a trade-off:
  - **Step 10, the hybrid,** ties step 7 + filter on boosting AUC (0.717 vs 0.716) with the smallest spread between seeds. It samples about 3 times faster than pure diffusion, because it only runs the last 20 or 40 of the 100 denoising steps. Its noise level was chosen on the selection slice: 200 of 1,000 steps for one seed and 400 for the other two.
  - **What the GAN start buys (control run, same step 8 denoiser and filter):**
    - At the same 20 or 40 denoiser calls, pure diffusion from noise has a higher forest AUC in every seed, by 0.016 to 0.027. On boosting AUC the gap is −0.003 to +0.021 and not consistent.
    - Step 10 matches step 8 (100 calls) on boosting AUC (0.717 vs 0.718). Its forest AUC is 0.008 to 0.016 lower in all three seeds. That gap is the size of cross-CPU rerun noise, so with three seeds it is unclear whether the GAN start also improves quality. What it clearly buys is speed: 20 to 40 calls instead of 100.
    - A GAN start without domain rules (step 4) matches step 5 on detection AUC, but its utility is about 0.01 lower.
    - A start of independently drawn columns gives no gain over pure diffusion. The selection slice picks the highest noise level tried (800) in every seed, and at 80 calls it matches pure diffusion with 80 calls.
    - The pure-diffusion comparison uses the same strided ancestral sampler as step 8. A better few-step sampler (for example a deterministic one) was not tried and could narrow the gap.
    - Details: `results/ablations/SUMMARY.md`.
  - **Step 9, latent diffusion,** has the best forest score and the best utility, but the largest spread between seeds. It also drops rare categories: category coverage is 0.89, against 0.99 for real rows. A check on the saved models locates the loss in the VAE decoder, not in the diffusion model over latents.
    - Real rows passed through the VAE alone, with no diffusion, lose almost exactly the same categories.
    - The lost categories are never the most likely category in any row (highest probability about 0.3), so argmax decoding never emits them.
    - Sampling from the decoder's softmax instead raises coverage on the selection slice from 0.90 to 0.98, at the same selection AUC. It costs some row accuracy, though: native-country is reconstructed correctly 96% of the time instead of 98%, and twice as many rows break a rule before the filter.
  - **Step 8** is the simplest model that meets every target.
- **Without rules, diffusion breaks them often.** 23% of step 7's rows break a domain rule, against 7% for the comparable GAN step.
  - The rule filter alone fixes realism (0.764 → 0.711).
  - Adding the rule penalty in training (step 8) brings the share of rows the filter has to redraw from 23% to 0.02–0.04%, at the same realism.
  - For the GAN, the penalty also improves realism: the filter alone gives 0.767, and the penalty with the filter gives 0.749.
- **Speed:** a GAN draws 1,000 rows in about 0.04 seconds on one CPU thread. The diffusion steps take 2.2 to 2.7 seconds, or 3.5 for step 7 + filter, which redraws 23% of its rows. Step 10 takes 0.9 seconds.
- **Copies are rarer than in real data, and the distance test shows no memorization.** Every row generated during evaluation was checked against every fit-slice row: about 127,000 rows per model, from three draws and three seeds.
  - Exact copies in all 13 columns: none for any GAN step or for steps 7, 8, 9 and 10, and one for step 7 + filter. For scale, 0.11% of unseen real selection rows match a fit row, because Adult contains repeated rows.
  - `fnlwgt` is close to unique per person, so the 12 other columns were also checked. 9.5 to 12.2% of diffusion rows and 9.7% of step 5 rows match a fit row there, against 15.9% of unseen real rows.
  - Step 0 cannot match exactly by construction, because it does not round numbers.
  - The share of rows closer to training rows stays close to 0.5 for every model.
  - For steps 7 + filter and 8 it is 0.514 to 0.517 in two of the three seeds, which is about two standard errors above 0.5. This slight signal is a question for the planned membership-inference study, not a finding yet.
- **Snapping capital values adds nothing measurable** (0.747 vs 0.749), so the GAN keeps step 5, where everything except the domain rules is learned.

## Method

### Data and splits

- **Dataset:** UCI Adult, 32,561 rows used for training. `education` is dropped and `education-num` is kept. `?` stays as its own category.
- **Splits:** for each seed (42, 68, 124) the 32,561 rows are split into a fit slice (20,838 rows), a selection slice (5,210) and an internal test split (6,513).
- **What each slice is for:**
  - Encoders and models are fit on the fit slice only.
  - Checkpoints, the snap threshold and step 10's noise level are chosen on the selection slice.
  - The internal test split was used to study which rows the detector catches. Because of that, the results above use the separate official test file.

### The GAN (step 5)

- **Conditioning:** a conditional WGAN-GP on income. The critic judges packs of 10 rows (PacGAN), and the generator emits straight-through Gumbel-softmax one-hots, so its categories are exact 0/1 like real rows.
- **Numeric columns:** each is encoded as a Bayesian Gaussian-mixture mode plus a position inside that mode, as in CTGAN. `capital-gain` and `capital-loss` add a zero/non-zero gate. Integer columns are rounded.
- **Training-by-sampling:** training conditions on categories sampled by log-frequency, so rare categories are learned.
- **Domain rules** come from what the fields mean, not from fitted numbers:
  - a husband is a married man, and a wife is a married woman;
  - occupation is missing exactly when workclass is missing or "Never-worked".

  In all 48,842 Adult rows the last two rules have no exceptions and the first has 4. The generator is trained with a penalty equal to the expected number of broken rules, computed from its category probabilities. At sampling time, any row that still breaks a rule is redrawn for the same label.
- **Checkpoint selection:** the kept epoch has zero hard-validity violations and the lowest cross-validated detection AUC on the selection slice.

### The diffusion models (steps 7 to 10)

- **A diffusion model learns to remove noise.** Training corrupts real rows step by step until they are pure noise, and teaches a network to undo one step. Generation starts from noise and denoises it into a new row. The network is conditioned on income.
- **Steps 7 and 8 (TabDDPM style):**
  - Numeric columns are mapped to a normal scale with a quantile transform and get Gaussian noise. The capital columns keep the zero/non-zero gate.
  - Categorical columns get multinomial noise, where each step has a small chance of replacing a category with a random one.
  - An MLP predicts the Gaussian noise and the clean categories. It is trained for 30,000 steps with an EMA of the weights, and sampling uses 100 of the 1,000 noise levels.
  - Step 8 adds the same rule penalty as the GAN, computed from the predicted clean categories, and the same rule filter.
- **Step 9 (TabSyn style):** a small transformer VAE turns each row into one 4-number vector per column. An EDM diffusion model learns those vectors and samples them with a 50-step deterministic solver. The rule penalty passes through the frozen decoder.
- **Step 10 (hybrid):** a step 5 GAN row is noised to level t0 and denoised by step 8 from there. t0 is chosen on the selection slice from 25, 50, 100, 200 and 400 out of 1,000. A control run widened the grid to 800, and the choice did not change.
- **Checkpoint selection:** the same rule as the GAN, scored every 3,000 training steps.

### Evaluation

- **Detection:** a random forest (200 trees) and a gradient-boosting classifier are each trained on selection rows vs. one synthetic draw. They are then scored on the test-file rows vs. a fresh draw.
- **Utility:** one random forest is trained on real fit rows and another on an equal number of synthetic rows. Both are scored on the test file. The ratio is `synthetic AUC / real AUC`.
- **Fidelity:** per-column KS for numeric columns, total variation for categorical columns, and category coverage.
- **Memorization:** each of 5,000 synthetic rows is compared with its nearest training row and its nearest test-file row, on standardized features. The table shows the share of synthetic rows that sit closer to the training rows. About 0.5 means the model generalizes rather than copies.
- **Copies:** every generated row is compared with every fit-slice row, on all 13 columns and on the 12 columns without `fnlwgt` (`ablations.py copies`).

## Repository structure

```
.
|-- ladder.py               # GAN steps 0-5: generator, training, domain rules, load_generator
|-- diffusion.py            # Diffusion steps 7-9: TabDDPM-style and latent (TabSyn-style) models, load_diffusion
|-- evaluate_holdout.py     # Scores every model on the official test file; builds steps 6 and 10
|-- ablations.py            # Control runs on saved models: exact copies, step 9 decoding, step 10 start
|-- summarize_ablations.py  # Builds results/ablations/SUMMARY.md
|-- summarize_holdout.py    # Builds results/holdout/SUMMARY.md and the figure
|-- sdv_baselines.py        # SDV reference generators on the internal test split
|-- summarize_ladder.py     # Summary of the internal test split results
|-- evaluator.py            # Metrics: hard validity, KS / TV, coverage, detection protocols, utility
|-- main.py                 # Data loading, leakage-free splits, baseline GAN / cGAN
|-- type_aware.py, run_stage2.py, eval_legacy_selection.py, experiments.ipynb   # baseline GAN / cGAN study
|-- tests/                  # Unit tests for splits, leakage, selection, metrics and the diffusion math
`-- results/
    |-- holdout/            # Main results: per-run JSON, SUMMARY.md, holdout_results.png
    |-- ablations/          # Control runs: exact copies, step 9 decoding, step 10 start
    |-- ladder/             # Internal test split results for steps 0-9 and SDV
    `-- legacy/, stage1/, stage2/, metrics_schema.json   # baseline study records
```

## How to run

You need two data files, both public at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult). Neither is committed (`data/` is gitignored).

- `data/adult.arff`: the 32,561 training rows, in ARFF format.
- `data/adult_test.csv`: the official `adult.test` file, either as published or as a plain CSV.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python ladder.py                 # GAN steps 0-5 on three seeds (CPU, about 1 hour per step-4/5 run)
python diffusion.py              # diffusion steps 7-9 on three seeds (CPU, 45 minutes per step-7/8 run, 1.5 hours per step-9 run)
python evaluate_holdout.py       # score everything on the test file (fits SDV models if missing)
python summarize_holdout.py      # rebuild the table and figure
python ablations.py copies       # control runs on the saved models (also: decoding, hybrid)
python summarize_ablations.py    # rebuild results/ablations/SUMMARY.md
```

`requirements.txt` pins `torch==2.14.1+cpu`. If pip cannot find that build, use the PyTorch CPU wheel index (`--index-url https://download.pytorch.org/whl/cpu`).

### Reusing a trained model

Every run saves its selected model to `models/ladder/{step}_seed{seed}.pt`. These files are not in the repository. To draw new rows without retraining:

```python
from ladder import load_generator
from diffusion import load_diffusion

sample = load_generator("models/ladder/5_rules_seed42.pt")        # GAN
sample = load_diffusion("models/ladder/8_ddpm_rules_seed42.pt")   # diffusion
rows = sample([0] * 760 + [1] * 240, seed=0)   # income labels in, decoded DataFrame out
```

The GAN checkpoints of steps 3 and 5 and all diffusion checkpoints also keep what a membership-inference study needs, so it can run without retraining:

- the critic, or the EMA and raw diffusion weights;
- final weights and optimizer states, to resume training;
- four weight snapshots over training;
- the loss curve, the encoders and the noise schedule;
- the row indexes of the fit slice, so it is known which rows were members.

`GaussianMultinomialDiffusion.per_row_loss` and `EDMDiffusion.per_row_loss` give each row's denoising error at a chosen noise level.

## Limitations

- One table, three seeds and fixed hyperparameters per step. There was no tuning, and the penalty weight is fixed at 1.
- Steps were added in one fixed order, so effects that depend on order are not measured.
- The diffusion models were still improving at the last training step, which is also where the learning rate reaches zero. Longer schedules were not tried.
- Latent diffusion (step 9) misses some rare categories, because its VAE decoder never gives them the top probability and decoding takes the argmax. Steps 8 and 10 miss a few (coverage 0.89, 0.94 and 0.95, against 0.99 for real rows).
- The official test file is a reused benchmark. No checkpoint, threshold or noise level was selected on it, but which steps to build, and their grids and hyperparameters, were decided after seeing earlier results on it.
- Three seeds per model, and the same seed differs by about 0.01 AUC across CPUs. Gaps of 0.01 to 0.02 are consistent in direction at best, not established.
- The domain rules are specific to Adult. They are kept in one function (`rule_violations`), separate from the models.
- A detector still reaches about 0.70, so the synthetic rows are not indistinguishable from real ones.
- The memorization check is a distance test, not a formal privacy guarantee.

## What I would try next

- Membership-inference attacks on the saved models (distance-based, critic score and denoising loss), with a focus on rare individuals, and DP-SGD training to measure the privacy-utility trade-off.
- Decode step 9 by sampling from the softmax, and recover the few rare categories steps 8 and 10 miss, for example with training-by-sampling as in the GAN.
- Combine this with my face-verification work: conditional generation to balance under-represented groups, measured by its effect on verification error.

---

Author: Dor Meir

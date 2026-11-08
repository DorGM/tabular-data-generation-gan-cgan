# Legacy baseline inventory

Captured 2026-10-14 before any model change. These files record the original Adult GAN / cGAN results. They were read, not retrained. Checkpoint weights were not modified.

Pre-edit file hashes are in `source_hashes.json`. `main.py` was not edited; its SHA-256 is still `deeeb8ec18eca6f042711a46f1b5937c957f7491c41d960003187aee0e07d4c3`.

## Provenance

| Artifact | Role |
| --- | --- |
| `selection_age_means.json` | Full 5,210-row selection age means recomputed from `data/adult.arff` |
| `checkpoint_records.json` | Selection history copied out of the 17 summary checkpoints |
| `run_manifest.json` | One row per completed run |
| `source_hashes.json` | Hashes, config fingerprints, and the environment used to read the baseline |
| `checkpoints/` | Original weights (not included in this repository). 17 summary files plus 17 epoch directories of 12 state dicts each |

The `data/adult.arff` file used for these runs has SHA-256 `504fd9dc0a28d56b20ad0ef5d54f488570c9523f6e3f26d79c1093bc199cc940` (3,975,854 bytes). It is not included in this repository.

## Environment

Python 3.13.2, torch 2.14.1+cpu, numpy 2.3.0, pandas 2.3.0, scipy 1.15.3, scikit-learn 1.7.0, matplotlib 3.10.3, CUDA unavailable. This matches the notebook stdout from 2026-10-13 and `requirements.txt`.

## Config fingerprints

SHA-256 of the canonical JSON config (Adam, latent 64, hidden 128, batch 64, dropout 0, weight decay 0, 120 epochs):

| Config | Fingerprint |
| --- | --- |
| sigmoid, lr 1e-3 | `9d0345462e1c812bebcc85f6e133249020ed7a62dfe39c601a41a4bb162912c8` |
| linear, lr 1e-3 | `533337c897108c7711ae77a2aa5d2f5cca8869f551ecd6e1c77c5af367ead5ec` |
| sigmoid, lr 1e-4 | `1fce04d183811320fc514a67e3573750a324c78096f9ede38560163ed5782ae4` |
| linear, lr 1e-4 | `20806b515e33f75c63cc59f4104cc0d4ec7589751bc9f0cec667dce7cf75e398` |

## Selection-slice age means

Recomputed with `prepare_experiment` and `decode_rows` on every selection row. Linear and sigmoid agree at four decimal places. The old 39.00 / 33.00 / 39.00 figures are the first 8 rows.

| Seed | Rows | Full-slice age mean | First 8 rows |
| ---: | ---: | ---: | ---: |
| 42 | 5,210 | 38.6503 | 39.00 |
| 68 | 5,210 | 38.6188 | 33.00 |
| 124 | 5,210 | 38.3438 | 39.00 |

Fit / selection / test sizes are 20,838 / 5,210 / 6,513, with 92 encoded columns. Income rates on seed 42 are 0.240810 (fit), 0.240883 (selection), and 0.240749 (test). The same rates were obtained for seeds 68 and 124.

## Original negative findings

These numbers come from the 2026-10-13 notebook and the saved checkpoint histories. The audit did not rescore the held-out test.

Primary lr `1e-3`: every linear checkpoint failed the plausible-numeric gate. Accepted sigmoid runs have held-out detection AUC from 0.999987 to 1.0. Accepted cGAN sigmoid utility ratios are 0.846179 (seed 42: real 0.910487, synthetic 0.770435) and 0.833801 (seed 124: real 0.909893, synthetic 0.758670). `cgan_sigmoid` seed 68 was not accepted.

The lr `1e-4` follow-up accepts `cgan_linear` seeds 42 and 124 and `cgan_sigmoid` seed 68. Their held-out detection AUC is 1.0. The recovered seed-68 efficacy ratio is 0.596872.

These are not successful fidelity results.

## Checkpoint status

Every summary file has 120 logged loss values, so none of the 17 runs stopped early. An acceptable file's `state_dict` is the selected epoch, because training restores that state before the notebook saves. An ineligible file has no selected state; its weights are the last epoch and are not a chosen model. Epoch files are raw generator state dicts, without the config wrapper.

Training cannot resume from the middle of an epoch. A finished summary file can be reloaded. New runs must use a different checkpoint directory.

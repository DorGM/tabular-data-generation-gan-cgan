# Results on the official Adult test file (16,281 rows no model or rule has seen)

Mean ± sample SD over split seeds 42, 68 and 124. Detectors are fit on selection rows vs. a synthetic
draw and scored on holdout rows vs. a fresh draw. The reference row uses real rows the models never
trained on, so it shows the best a generator could score.

| Model | Detection AUC, forest ↓ | Detection AUC, boosting ↓ | Utility ratio ↑ | Numeric KS ↓ | Categorical TV ↓ | Rule-breaking rows ↓ | Closer to training rows (≈0.5 = no memorization) | Exact copies of training rows | Sampling seconds per 1,000 rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Real rows (unseen), reference | 0.499 ± 0.005 | 0.504 ± 0.004 | 1.000 ± 0.004 | 0.015 ± 0.001 | 0.018 ± 0.002 | 0.000 ± 0.000 | 0.501 ± 0.007 | 0.001 ± 0.000 | n/a |
| 0 Baseline | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.617 ± 0.256 | 0.737 ± 0.182 | 0.515 ± 0.276 | 0.417 ± 0.519 | 0.607 ± 0.356 | 0.000 ± 0.000 | n/a |
| 1 + Sampling, integers, real zeros, base-rate labels | 0.974 ± 0.010 | 0.981 ± 0.008 | 0.842 ± 0.075 | 0.154 ± 0.025 | 0.167 ± 0.103 | 0.002 ± 0.002 | 0.508 ± 0.004 | 0.000 ± 0.000 | n/a |
| 2 + Straight-through one-hots | 0.947 ± 0.006 | 0.965 ± 0.004 | 0.910 ± 0.028 | 0.140 ± 0.017 | 0.079 ± 0.005 | 0.002 ± 0.001 | 0.501 ± 0.020 | 0.000 ± 0.000 | n/a |
| 3 + WGAN-GP and PacGAN | 0.936 ± 0.005 | 0.960 ± 0.003 | 0.952 ± 0.005 | 0.109 ± 0.008 | 0.019 ± 0.003 | 0.072 ± 0.022 | 0.492 ± 0.005 | 0.000 ± 0.000 | 0.007 ± 0.000 |
| 4 + Mode-specific normalization, training-by-sampling | 0.781 ± 0.005 | 0.826 ± 0.007 | 0.945 ± 0.018 | 0.029 ± 0.002 | 0.027 ± 0.003 | 0.071 ± 0.011 | 0.498 ± 0.010 | 0.000 ± 0.000 | n/a |
| 4 + rule filter at sampling only | 0.767 ± 0.008 | 0.822 ± 0.010 | 0.944 ± 0.018 | 0.030 ± 0.001 | 0.032 ± 0.003 | 0.000 ± 0.000 | 0.499 ± 0.011 | 0.000 ± 0.000 | n/a |
| 5 + Domain-rule loss penalty and rule filter | 0.749 ± 0.004 | 0.808 ± 0.006 | 0.960 ± 0.013 | 0.041 ± 0.001 | 0.040 ± 0.003 | 0.000 ± 0.000 | 0.501 ± 0.003 | 0.000 ± 0.000 | 0.044 ± 0.002 |
| 6 + Capital values snapped to common real values | 0.747 ± 0.004 | 0.807 ± 0.003 | 0.961 ± 0.013 | 0.040 ± 0.001 | 0.040 ± 0.003 | 0.000 ± 0.000 | 0.500 ± 0.003 | 0.000 ± 0.000 | 0.043 ± 0.001 |
| 7 Diffusion (TabDDPM style), no rules | 0.764 ± 0.015 | 0.773 ± 0.012 | 0.982 ± 0.002 | 0.015 ± 0.001 | 0.020 ± 0.002 | 0.230 ± 0.017 | 0.502 ± 0.009 | 0.000 ± 0.000 | 2.679 ± 0.179 |
| 7 + rule filter at sampling only | 0.711 ± 0.010 | 0.716 ± 0.013 | 0.981 ± 0.002 | 0.020 ± 0.002 | 0.044 ± 0.005 | 0.000 ± 0.000 | 0.509 ± 0.012 | 0.000 ± 0.000 | 3.473 ± 0.088 |
| 8 Diffusion + domain-rule loss and rule filter | 0.716 ± 0.012 | 0.718 ± 0.013 | 0.963 ± 0.006 | 0.016 ± 0.001 | 0.039 ± 0.003 | 0.000 ± 0.000 | 0.509 ± 0.012 | 0.000 ± 0.000 | 2.638 ± 0.191 |
| 9 Latent diffusion (TabSyn style) + rules | 0.700 ± 0.028 | 0.732 ± 0.028 | 0.983 ± 0.006 | 0.030 ± 0.010 | 0.046 ± 0.008 | 0.000 ± 0.000 | 0.500 ± 0.011 | 0.000 ± 0.000 | 2.247 ± 0.213 |
| 10 Step 5 GAN rows refined by step 8 diffusion | 0.705 ± 0.008 | 0.717 ± 0.009 | 0.980 ± 0.004 | 0.026 ± 0.004 | 0.040 ± 0.005 | 0.000 ± 0.000 | 0.495 ± 0.007 | 0.000 ± 0.000 | 0.924 ± 0.328 |
| SDV Gaussian copula | 0.998 ± 0.001 | 0.999 ± 0.001 | 0.868 ± 0.032 | 0.326 ± 0.041 | 0.010 ± 0.002 | 0.332 ± 0.042 | 0.484 ± 0.012 | 0.000 ± 0.000 | n/a |
| SDV TVAE | 0.937 ± 0.012 | 0.950 ± 0.009 | 0.964 ± 0.008 | 0.161 ± 0.019 | 0.071 ± 0.010 | 0.030 ± 0.007 | 0.491 ± 0.011 | 0.000 ± 0.000 | n/a |
| SDV CTGAN | 0.935 ± 0.019 | 0.953 ± 0.013 | 0.967 ± 0.007 | 0.168 ± 0.021 | 0.100 ± 0.005 | 0.081 ± 0.010 | 0.519 ± 0.020 | 0.000 ± 0.000 | n/a |
| SDV CTGAN + rule filter | 0.933 ± 0.020 | 0.954 ± 0.013 | 0.968 ± 0.007 | 0.169 ± 0.023 | 0.097 ± 0.006 | 0.000 ± 0.000 | 0.516 ± 0.022 | 0.000 ± 0.000 | n/a |

Step 10 noise level t0 (of 1,000) chosen on the selection slice, per seed: [400, 400, 200].

Snap threshold k chosen on the selection slice, per seed: [20, 50, 10].

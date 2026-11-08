# Improvement ladder: results on the held-out test split

Mean ± sample SD over split seeds 42, 68 and 124. Each rung adds one change to the rung above it.
Checkpoints were chosen on the selection slice; the test split was scored once per run.

| Model | Seeds | Detection AUC ↓ | Utility ratio ↑ | Numeric KS ↓ | Categorical TV ↓ | Coverage ↑ | Hard-validity violations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 Baseline (published design) | 3 | 1.000 ± 0.000 | 0.615 ± 0.260 | 0.736 ± 0.183 | 0.513 ± 0.277 | 0.315 ± 0.123 | 6513 |
| 1 + Sampling, integers, real zeros, base-rate labels | 3 | 0.975 ± 0.009 | 0.845 ± 0.065 | 0.157 ± 0.024 | 0.168 ± 0.103 | 0.592 ± 0.083 | 0 |
| 2 + Straight-through one-hots | 3 | 0.947 ± 0.004 | 0.906 ± 0.023 | 0.143 ± 0.016 | 0.079 ± 0.008 | 0.844 ± 0.022 | 0 |
| 3 + WGAN-GP and PacGAN | 3 | 0.938 ± 0.006 | 0.951 ± 0.005 | 0.112 ± 0.005 | 0.023 ± 0.004 | 0.944 ± 0.002 | 0 |
| 4 + Mode-specific normalization, training-by-sampling | 3 | 0.779 ± 0.005 | 0.944 ± 0.019 | 0.031 ± 0.006 | 0.027 ± 0.002 | 0.994 ± 0.005 | 0 |
| 5 + Domain-rule loss penalty and rule filter | 3 | 0.744 ± 0.005 | 0.963 ± 0.004 | 0.036 ± 0.001 | 0.041 ± 0.010 | 0.982 ± 0.005 | 0 |
| SDV Gaussian copula | 3 | 0.998 ± 0.001 | 0.873 ± 0.021 | 0.328 ± 0.040 | 0.015 ± 0.000 | 0.975 ± 0.008 | 0 |
| SDV TVAE | 3 | 0.938 ± 0.012 | 0.962 ± 0.004 | 0.164 ± 0.016 | 0.073 ± 0.009 | 0.792 ± 0.015 | 0 |
| SDV CTGAN | 3 | 0.934 ± 0.018 | 0.962 ± 0.008 | 0.168 ± 0.018 | 0.102 ± 0.006 | 1.000 ± 0.000 | 0 |

# Follow-up checks on the Adult benchmark

Built by `summarize_ablations.py` from `ablations.py` runs on the saved checkpoints (no retraining). Mean ± SD over the three split seeds.

## Copies of training rows over every generated row

Every row of the three evaluation draws (selection-sized, test-sized and fit-sized), for all three seeds, compared with every fit-slice row. "All 13 columns" is an exact copy. Because `fnlwgt` is close to unique per person, the 12 other columns are also checked (a quasi-identifier match). Step 0 does not round numbers, so it cannot match exactly by construction.

Scale from real data: of 15,630 real selection-slice rows that no model trained on, 17 (0.11%) equal a fit row in all 13 columns and 2,483 (15.9%) in the 12 columns without `fnlwgt`.

| Model | Rows checked | Copies, all 13 columns | Matches without fnlwgt | Draw rebuilt exactly |
| --- | ---: | ---: | ---: | :---: |
| 0_baseline | 126,987 | 0 (0.000%) | 0 (0.0%) | yes |
| 1_sampling | 126,987 | 0 (0.000%) | 3,619 (2.8%) | yes |
| 2_st_onehot | 126,987 | 0 (0.000%) | 2,815 (2.2%) | yes |
| 3_wgan_gp | 126,987 | 0 (0.000%) | 1,254 (1.0%) | yes |
| 4_ctgan_norm | 126,987 | 0 (0.000%) | 10,690 (8.4%) | yes |
| 4_ctgan_norm+reject | 126,987 | 0 (0.000%) | 11,523 (9.1%) | yes |
| 5_rules | 126,987 | 0 (0.000%) | 12,295 (9.7%) | yes |
| 6_rules_snap | 126,987 | 0 (0.000%) | 12,309 (9.7%) | yes |
| 7_ddpm | 126,987 | 0 (0.000%) | 12,111 (9.5%) | yes |
| 7_ddpm+reject | 126,987 | 1 (0.001%) | 15,511 (12.2%) | yes |
| 8_ddpm_rules | 126,987 | 0 (0.000%) | 14,953 (11.8%) | yes |
| 9_latent_rules | 126,987 | 0 (0.000%) | 12,020 (9.5%) | yes |
| 10_gan_refine | 126,987 | 0 (0.000%) | 15,172 (11.9%) | yes |
| sdv_gaussian_copula | 126,987 | 0 (0.000%) | 0 (0.0%) | yes |
| sdv_tvae | 126,987 | 2 (0.002%) | 12,975 (10.2%) | yes |
| sdv_ctgan | 126,987 | 1 (0.001%) | 2,755 (2.2%) | yes |
| sdv_ctgan+reject | 126,987 | 1 (0.001%) | 3,050 (2.4%) | yes |

## Step 9: where rare categories are lost

Coverage = share of categories present in the real rows that appear at least once; rare recall = the same for categories under 1% frequency, measured against the real rows of the same slice.

| Rows | Decoding | Coverage | Rare recall | Categorical TV | Selection AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| Generated latents, with filter | argmax (current) | 0.899 ± 0.009 | 0.500 ± 0.012 | 0.043 ± 0.010 | 0.6983 ± 0.0272 |
| Generated latents, with filter | softmax sample | 0.980 ± 0.004 | 0.953 ± 0.023 | 0.043 ± 0.011 | 0.6999 ± 0.0247 |
| Real fit rows through the VAE only | argmax | 0.897 ± 0.011 | 0.486 ± 0.025 | 0.003 ± 0.000 | - |
| Real fit rows through the VAE only | softmax sample | 0.990 ± 0.003 | 0.964 ± 0.013 | 0.001 ± 0.000 | - |
| Real selection rows through the VAE only | argmax | 0.902 ± 0.014 | 0.492 ± 0.024 | 0.003 ± 0.000 | - |
| Real selection rows through the VAE only | softmax sample | 0.970 ± 0.009 | 0.922 ± 0.035 | 0.002 ± 0.000 | - |
| Step 7 (reference) | - | 0.981 ± 0.018 | 0.953 ± 0.042 | 0.024 ± 0.002 | - |
| Step 8 (reference) | - | 0.922 ± 0.008 | 0.945 ± 0.013 | 0.041 ± 0.003 | - |

In seed42, argmax never emits 24 categories that occur in the fit slice: none of them is the most likely category in any generated row (highest probability 0.28). The 12 most frequent are shown:

| Column | Category | Fit share | Mean softmax probability | Highest probability in any row |
| --- | --- | ---: | ---: | ---: |
| native-country | Columbia | 0.0019 | 0.0015 | 0.244 |
| native-country | Taiwan | 0.0014 | 0.0011 | 0.192 |
| native-country | Haiti | 0.0012 | 0.0065 | 0.282 |
| native-country | Portugal | 0.0012 | 0.0011 | 0.167 |
| native-country | Iran | 0.0011 | 0.0011 | 0.170 |
| native-country | France | 0.0011 | 0.0007 | 0.121 |
| native-country | Nicaragua | 0.0010 | 0.0007 | 0.109 |
| native-country | Greece | 0.0009 | 0.0007 | 0.119 |
| native-country | Peru | 0.0009 | 0.0006 | 0.091 |
| native-country | Ireland | 0.0008 | 0.0008 | 0.151 |
| native-country | Cambodia | 0.0006 | 0.0005 | 0.078 |
| native-country | Laos | 0.0006 | 0.0007 | 0.085 |

## Step 10: does the GAN start matter?

All rows use the same step 8 denoiser and rule filter. For the start variants, t0 is chosen on the selection slice per seed. Denoiser calls = t0 / 10. Test-file numbers (a reused benchmark).

| Variant | t0 per seed | Denoiser calls | Forest AUC | Boosting AUC | Utility | Coverage |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Step 10 as published: GAN 5 start, t0 grid up to 400 | 400, 200, 400 | 40, 20, 40 | 0.705 ± 0.008 | 0.717 ± 0.009 | 0.980 ± 0.004 | 0.951 ± 0.014 |
| GAN step 5 start, t0 grid up to 800 | 400, 200, 400 | 40, 20, 40 | 0.705 ± 0.008 | 0.717 ± 0.009 | 0.980 ± 0.004 | 0.951 ± 0.014 |
| GAN step 4 start (no rules), t0 grid up to 800 | 400, 400, 400 | 40, 40, 40 | 0.707 ± 0.005 | 0.712 ± 0.008 | 0.970 ± 0.002 | 0.935 ± 0.000 |
| Independent-marginals start, t0 grid up to 800 | 800, 800, 800 | 80, 80, 80 | 0.716 ± 0.011 | 0.718 ± 0.015 | 0.964 ± 0.003 | 0.950 ± 0.013 |
| Pure diffusion, 20 steps | -, -, - | 20, 20, 20 | 0.730 ± 0.011 | 0.737 ± 0.011 | 0.965 ± 0.006 | 0.935 ± 0.000 |
| Pure diffusion, 40 steps | -, -, - | 40, 40, 40 | 0.723 ± 0.011 | 0.722 ± 0.015 | 0.967 ± 0.008 | 0.935 ± 0.000 |
| Pure diffusion, 60 steps | -, -, - | 60, 60, 60 | 0.720 ± 0.013 | 0.720 ± 0.014 | 0.968 ± 0.009 | 0.931 ± 0.005 |
| Pure diffusion, 80 steps | -, -, - | 80, 80, 80 | 0.719 ± 0.013 | 0.721 ± 0.013 | 0.970 ± 0.009 | 0.943 ± 0.014 |
| Pure diffusion, 100 steps (step 8) | -, -, - | 100, 100, 100 | 0.716 ± 0.012 | 0.718 ± 0.013 | 0.963 ± 0.006 | 0.943 ± 0.014 |

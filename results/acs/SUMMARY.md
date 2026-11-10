# Fresh evaluation on ACS California 2018 (ACSIncome)

One-time evaluation on the reserved test part (30,000 people, households kept together), opened 2026-11-10T07:50:17Z. Protocol: `PROTOCOL.md`. Every model was fit on the training part (30,000 people) and every choice was made on the validation part (8,000). Mean ± SD over training seeds 42, 68 and 124 on one fixed split.

| Model | Forest AUC ↓ | Boosting AUC ↓ | Utility ↑ | KS ↓ | TV ↓ | Coverage ↑ | Rare recall ↑ | Occupation coverage | Birthplace coverage | Rule-breaking rows | Filter redraws | Closer to training (0.5 = none) | Exact copies / rows | Denoiser calls | Seconds per 1k rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Real reference rows (best reachable) | 0.502 ± 0.001 | 0.500 ± 0.003 | 1.000 ± 0.001 | 0.005 ± 0.001 | 0.017 ± 0.000 | 0.987 ± 0.006 | 0.980 ± 0.001 | 0.991 | 0.957 | 0.00 ± 0.00% | - | 0.491 ± 0.015 | 2406 / 204,000 | -, -, - | - |
| 5 GAN + rules | 0.743 ± 0.009 | 0.774 ± 0.012 | 0.969 ± 0.004 | 0.061 ± 0.005 | 0.057 ± 0.002 | 0.988 ± 0.005 | 0.948 ± 0.015 | 0.945 | 0.959 | 0.00 ± 0.00% | 0.087 ± 0.039% | 0.505 ± 0.019 | 844 / 204,000 | -, -, - | 0.08 ± 0.01 |
| 7 Diffusion, no rules, no filter | 0.773 ± 0.006 | 0.770 ± 0.004 | 0.983 ± 0.001 | 0.019 ± 0.003 | 0.023 ± 0.001 | 0.993 ± 0.002 | 0.974 ± 0.006 | 0.976 | 0.967 | 10.32 ± 0.21% | - | 0.511 ± 0.007 | 382 / 204,000 | 100, 100, 100 | 8.08 ± 0.47 |
| 7 Diffusion + rule filter | 0.734 ± 0.006 | 0.738 ± 0.004 | 0.982 ± 0.001 | 0.016 ± 0.003 | 0.037 ± 0.001 | 0.992 ± 0.002 | 0.970 ± 0.005 | 0.974 | 0.962 | 0.00 ± 0.00% | 10.324 ± 0.210% | 0.511 ± 0.007 | 439 / 204,000 | 100, 100, 100 | 8.50 ± 0.75 |
| 8 Diffusion + rule penalty + filter | 0.815 ± 0.005 | 0.817 ± 0.005 | 0.972 ± 0.011 | 0.013 ± 0.002 | 0.062 ± 0.002 | 0.991 ± 0.003 | 0.974 ± 0.007 | 0.976 | 0.971 | 0.00 ± 0.00% | 0.029 ± 0.008% | 0.508 ± 0.003 | 418 / 204,000 | 100, 100, 100 | 8.07 ± 0.17 |
| 9 Latent diffusion + rules | 0.722 ± 0.039 | 0.708 ± 0.049 | 0.961 ± 0.013 | 0.026 ± 0.009 | 0.054 ± 0.019 | 0.981 ± 0.006 | 0.957 ± 0.001 | 0.968 | 0.935 | 0.00 ± 0.00% | 0.207 ± 0.049% | 0.506 ± 0.004 | 379 / 204,000 | 99, 99, 99 | 2.28 ± 0.20 |
| 10 GAN rows refined by step 8 | 0.727 ± 0.013 | 0.715 ± 0.008 | 0.968 ± 0.003 | 0.058 ± 0.009 | 0.059 ± 0.001 | 0.988 ± 0.004 | 0.947 ± 0.012 | 0.944 | 0.959 | 0.00 ± 0.00% | 0.127 ± 0.034% | 0.506 ± 0.017 | 958 / 204,000 | 2, 2, 2 | 0.29 ± 0.00 |
| Control: independent-marginals start refined by step 8 | 0.815 ± 0.010 | 0.819 ± 0.011 | 0.979 ± 0.002 | 0.013 ± 0.001 | 0.058 ± 0.006 | 0.990 ± 0.003 | 0.975 ± 0.007 | 0.982 | 0.959 | 0.00 ± 0.00% | 0.989 ± 1.672% | 0.500 ± 0.003 | 351 / 204,000 | 80, 10, 80 | 4.56 ± 3.17 |
| Control: step 8 from noise, same calls as step 10 | 0.999 ± 0.000 | 0.999 ± 0.000 | 0.813 ± 0.035 | 0.480 ± 0.024 | 0.148 ± 0.032 | 0.896 ± 0.010 | 0.780 ± 0.047 | 0.820 | 0.704 | 0.00 ± 0.00% | 0.000 ± 0.000% | 0.522 ± 0.009 | 28 / 204,000 | 2, 2, 2 | 0.17 ± 0.00 |
| Control: step 8 from noise, 20 calls | 0.817 ± 0.005 | 0.820 ± 0.007 | 0.978 ± 0.009 | 0.028 ± 0.001 | 0.059 ± 0.001 | 0.985 ± 0.002 | 0.964 ± 0.006 | 0.972 | 0.945 | 0.00 ± 0.00% | 0.026 ± 0.019% | 0.506 ± 0.006 | 479 / 204,000 | 20, 20, 20 | 1.60 ± 0.05 |
| Control: step 8 from noise, 40 calls | 0.816 ± 0.007 | 0.815 ± 0.007 | 0.975 ± 0.009 | 0.018 ± 0.001 | 0.062 ± 0.002 | 0.987 ± 0.005 | 0.970 ± 0.005 | 0.974 | 0.959 | 0.00 ± 0.00% | 0.017 ± 0.009% | 0.510 ± 0.003 | 440 / 204,000 | 40, 40, 40 | 3.09 ± 0.33 |

Validation choices per seed (`results/acs/selection.json`):

- Seed 42: step 9 decoding **sample**, step 10 t0 **25**, marginals-start t0 **800**.
- Seed 68: step 9 decoding **sample**, step 10 t0 **25**, marginals-start t0 **100**.
- Seed 124: step 9 decoding **sample**, step 10 t0 **25**, marginals-start t0 **800**.

## Pre-registered hypotheses (PROTOCOL.md section 7)

Boosting AUC is primary. A difference counts only if the 3-seed means differ by at least 0.01.

1. **Diffusion beats the GAN.** Boosting AUC: 5 = 0.774, 7 + filter = 0.738, 8 = 0.817. Verdict: **not supported**.
2. **The rule penalty removes most rule-breaking.** Filter redraws: step 7 10.32%, step 8 0.029% (target under 0.1%). Boosting AUC of 8 minus 7 + filter: +0.079 (limit +0.01). Verdict: **not supported**.
3. **Latent diffusion keeps fewer categories.** Column-mean coverage: 9 = 0.981 (decoding chosen on validation: sample), lower of 7 + filter and 8 = 0.991. Verdict: **supported**.
4. **The GAN start helps.** Boosting AUC of step 10 = 0.715; pure diffusion at the same calls minus step 10: +0.285; marginals start minus step 10: +0.105 (forest: +0.271 and +0.088). Verdict on quality: **supported**.

Environment: Intel(R) Xeon(R) Processor @ 2.10GHz, torch 2.9.1+cu128, numpy 2.5.3, scikit-learn 1.9.1, 1 thread per run.

# Type-aware cGAN follow-up: selection comparison

Fresh type-aware cGAN, seeds 42, 68, and 124, 120 epochs, sigmoid-split data, BCE discriminator. Scored on the selection slice only (5,210 real rows and 5,210 synthetic rows, sample seed 42). The test split was not used. The comparison epoch is the predeclared rule: lowest violation fraction, then the mean of KS and categorical total variation, then protocol A AUC, then the earlier epoch.

| Seed | Epoch | Violations | KS mean | TV mean | Coverage | Rare recall | Protocol A AUC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 70 | 0 | 0.2019 | 0.3103 | 0.378 | 0.0000 | 0.9959 |
| 68 | 120 | 0 | 0.2466 | 0.2975 | 0.495 | 0.0465 | 0.9964 |
| 124 | 110 | 0 | 0.2429 | 0.1775 | 0.604 | 0.0233 | 0.9919 |

Every scored checkpoint, not only these three, had zero hard-validity violations. Protocol A AUC stayed above 0.99 throughout. Legacy sigmoid cGAN rows failed the integer check on all 5,210 selection rows and had higher KS and total variation. The type-aware model fixes that structural failure and improves those distances. It does not make the synthetic rows hard to detect.

An earlier seed-42 run stopped at epoch 40. It was not used for this comparison.

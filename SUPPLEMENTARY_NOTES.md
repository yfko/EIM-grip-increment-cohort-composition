# Supplementary Material

## Supplementary Note 1. Pre-registration (v1.1, 4 September 2026) and deviations

**Fixed design.** All 134 assessed adults; dominant-hand maximum of six grip trials as the primary outcome, non-dominant hand as sensitivity; demographic baseline sex, age, BMI; validity filter R > 0, X > 0, phase angle > 0; per-channel two-tailed 4 SD rule on ln(phase angle); one (R, X) pair per channel, phase angle not entered; primary EIM feature set the four forearm channels at 100 kHz (8 features); subject-level 10 × 5-fold cross-validation stratified on sex and the grip median with inner 5-fold tuning; Frisch–Waugh–Lovell residualization of outcome and features on demographics fitted within training folds; median imputation within folds; primary metric ΔMAE against the demographic baseline with 2,000-resample subject bootstrap; secondary ΔR² and AUC; seed 20260904.

**Primary hypothesis (H1).** Residualized forearm 100 kHz (R, X) features added to the demographic baseline reduce MAE with a 95% bootstrap interval whose lower bound exceeds zero; tested with the elastic-net model. Falsification: an interval including zero in the validation cohort changes the claim to "no demonstrable increment beyond demographics"; no substitution of site, frequency, metric or model is permitted.

**Pre-specified decision rules.** (a) No non-linear increment is claimed unless the XGBoost minus elastic-net ΔMAE interval excludes zero; an XGBoost advantage above 1.0 kg would change the claim to "value lies in non-linear interactions". (b) Any site or frequency other than forearm 100 kHz, and any multi-site model, is exploratory. (c) A bilateral mixed-effects model would be preferred only if its interval were narrower than the elastic net's.

**Model set.** Lasso on raw features plus demographics (literature protocol); elastic net on residualized features (H1); sparse-group Lasso over ten site-by-frequency groups (companion paper); PLS; XGBoost with and without monotone constraints; RBF kernel ridge; bilateral mixed-effects model; L2 logistic regression for AWGS weakness.

**Cohort-composition analysis.** Pre-registered: (i) train and test within the 109 functionally healthy adults; (ii) train on all 134 and evaluate the healthy subgroup out of fold; (iii) train on all 134 and evaluate the impaired subgroup. Added post hoc after the pre-registered analysis: (iv) train on the 109 only and apply to the 25 impaired adults.

**Development and validation sets.** Development: the 134 adults assessed by 4 September 2026, on which exploratory analyses had been run before pre-registration. Validation: adults recruited from 5 September 2026 at Taichung Municipal Geriatric Rehabilitation General Hospital, frozen on 31 December 2026 (target n ≈ 50), not accessed before freezing; H1 and the AUC analysis will be run once. Because the validation cohort is expected to be older and more impaired, a failure of H1 there will be interpreted against the demographic baseline's own R² in that cohort (a large shift in the baseline indicates population shift rather than model failure).

**Protocol change for the validation cohort.** Each site is recorded at both frequencies with one electrode application, the starting frequency alternating by participant number, so that frequency is no longer confounded with session order.

**Contrasts added at revision (round 1, in response to review).** Impaired-minus-healthy increment of the 134-trained model, subject bootstrap: 0.86 kg (−0.10 to 1.83), 7.2 percentage points of baseline error (−6.2 to 22.0). Training-cohort effect within the healthy subgroup, paired over the same 109 adults: −0.01 kg (−0.20 to 0.18). Minimum detectable subgroup difference at 80% power: ≈1.4 kg (bootstrap SE 0.51 kg). Null-feature check: forearm features permuted across participants 20 times gave increments of −0.05 kg (−0.20 to 0.09) in healthy adults, 0.02 kg (−0.16 to 0.32) in impaired adults and 0.01 kg (−0.18 to 0.26) in the post hoc cell. An earlier draft reported a "training-cohort interaction of 0.16 kg (−0.27 to 0.59)"; that figure was the all-134 increment minus the within-109 increment, not the healthy-subgroup contrast, and is withdrawn.

**Observed outcomes of the decision rules.** (a) XGBoost minus elastic net 0.13 kg (−0.14 to 0.39): no non-linear increment. (b) Sparse-group model selected forearm and erector spinae at 100 kHz in every fold and upper-arm and calf at 50 kHz in most folds; reported in the companion paper. (c) Bilateral model interval wider than the elastic net's (0.14, −0.27 to 0.44): not adopted.

**Deviations.** Tree models run with five rather than ten repeats; sparse-group inner grid 5 × 5 rather than 7 × 7 (software incompatibility); cell (iv) added post hoc. An exploratory pre-registration-era result, an apparent loss of the increment when training only on healthy adults (ΔR² −0.014 with a 24-feature Lasso including phase angle), did not survive the pre-registered elastic-net model (within-109 ΔMAE 0.33, 0.06 to 0.58) and is superseded.

## Supplementary Note 2. Age-specific reference limits do not identify grip-weak adults

Using the per-channel age-specific lower reference limits (fitted value minus 1.65 residual SD on the log phase angle) derived from the 109 functionally healthy adults in the companion reference-limit study, each participant's phase angle in each channel was compared with the limit at that participant's age. The proportion of channels below the limit was computed per participant.

| Group | n | Forearm channels below limit, mean | Participants with ≥ 1 forearm channel below limit |
|---|---|---|---|
| Functionally healthy (fitting cohort, in-sample) | 109 | 6.1% | 27% |
| Functionally impaired | 25 | 4.0% | 20% |
| Grip below AWGS threshold | 21 | 3.6% | 14% |

Mann–Whitney test, healthy versus impaired, forearm channels: p = 0.73. Across all 36 channels the corresponding proportions were 5.7% (healthy) and 5.9% (impaired), p = 0.34. In a sex-specific refit, lower-limb 100 kHz channels separated the groups (p = 0.0078 uncorrected over 16 comparisons) with the healthy group evaluated in-sample; this is reported as a lead only. The comparison is asymmetric (the healthy group is in-sample, the impaired group out-of-sample), which if anything favours finding a difference, so the null result for the forearm is conservative.

## Supplementary Note 3. Residual diagnostics for the demographic baseline

Ordinary least squares of dominant-hand grip on sex, age and BMI in all 134 adults (in-sample, for diagnostic purposes only; all reported performance is cross-validated). Residual SD 6.49 kg. Normality: Shapiro–Wilk W = 0.985, p = 0.16; skewness −0.01; excess kurtosis 0.86. Homoscedasticity: Breusch–Pagan test of squared residuals on fitted values, LM = 7.93, p = 0.005, so residual variance is not constant across the fitted range. Residual means by age band: under 40, −0.84 kg (SD 7.76); 40 to 64, +1.33 kg (SD 4.95); 65 and over, −0.26 kg (SD 6.02). Residual means by functional subgroup: healthy +1.69 kg; impaired −7.37 kg, the deviation from demographic expectation that defines the impaired subgroup. The heteroscedasticity is the expected consequence of that subgroup structure and is the reason the primary metric is the cross-validated MAE rather than an in-sample inference.

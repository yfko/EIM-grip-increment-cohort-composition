# EIM increment over demographics for grip prediction — analysis code

Analysis code, pre-registration and aggregate outputs for:

> Kao H-H, Ko Y-F. Small incremental value of electrical impedance myography over sex, age and
> body mass index for grip-strength prediction: a cross-validated study of 134 adults with and
> without functional impairment. *(submitted to Muscle & Nerve, September 2026)*

Forearm electrical impedance myography (EIM; resistance and reactance at 100 kHz, four channels)
was tested for what it adds to dominant-hand grip prediction beyond sex, age and body mass index
in 134 Taiwanese adults aged 18–87, of whom 109 were functionally healthy and 25 functionally
impaired. Features were residualized on demographics within training folds (Frisch–Waugh–Lovell),
models were evaluated by subject-level repeated cross-validation, and increments (ΔMAE) carry
subject-level bootstrap intervals. The code reproduces every number, table and figure in the
manuscript and the Supplementary Notes.

Ethics approval: Research Ethics Committee III, China Medical University and Hospital,
Taichung, Taiwan (**CMUH114-REC3-142**).

## Participant data are not in this repository

Individual participant data contain health information collected under the approved protocol
and are **not** publicly available. This repository holds code, the pre-registration, and
cohort-level aggregate outputs only. `.gitignore` excludes every `.csv`, `.xlsx` and `.npz`
(the `.npz` files written by `11_models.py` and `13_sgl_norms.py` hold one out-of-fold
prediction per participant). `check_no_participant_data.py` is run before every push.

## Layout

| Path | Content |
|---|---|
| `PREREGISTRATION_v1.1_zh-TW.md` | Pre-registration of 4 September 2026 (Chinese), including the protocol change for the validation cohort |
| `SUPPLEMENTARY_NOTES.md` | Supplementary Notes 1–3 as submitted (pre-registration summary and deviations; reference-limit projection; residual diagnostics) |
| `analysis/02_baseline.py`, `03_subgroup.py` | Demographic baseline and the panels A/B/C comparison (Table 2 footnote: sex alone, EIM alone) |
| `analysis/05_region_grid.py`, `06_region_grid_4sd.py` | Site × frequency grid, with and without the 4 SD rule (exploratory) |
| `analysis/07_lowerlimb_outcomes.py` | Gait and chair-rise outcomes (untestable; reported in `outputs/09`) |
| `analysis/11_models.py` | Pre-registered model suite: Lasso, elastic net (H1), sparse-group Lasso, PLS, XGBoost, kernel ridge, bilateral mixed model, logistic AUC (Table 2, Figure 2, Figure 4) |
| `analysis/13_sgl_norms.py` | Sparse-group Lasso group selection frequencies (companion methods paper) |
| `analysis/14_cohort_composition_h1model.py`, `16_train109_predict25.py` | Cohort-composition cells (i)–(iv) (Table 3, Figure 3) |
| `analysis/18_da_m2_check.py`, `19_subgroup_difference_bootstrap.py` | Permutation null-feature check; impaired-minus-healthy and training-cohort contrasts, MDE |
| `analysis/17_figures_clinical.py` | Figures 1–4 |
| `analysis/20_table1_cleaning.py` | Table 1 descriptives and data-cleaning counts |
| `analysis/projection_test.py`, `region_sex.py` | Supplementary Note 2 (age-specific reference-limit projection) |
| `outputs/` | Console outputs and result summaries of every script (aggregate values only; `11_models_runA/B.txt` and `12_模型組結果.md` hold the model-suite results) |
| `figures/` | Figures 1–4 (PNG) |

Scripts read the raw data from a local path that is not distributed; the cohort-membership file
comes from the companion reference-limit repository
(`EIM-dual-frequency-reference-limits`, `cohort_audit.csv`, likewise not distributed).

## Reproducing the results

```bash
pip install -r requirements.txt      # Python 3.9; XGBoost needs libomp on macOS (brew install libomp)
python analysis/11_models.py         # Table 2, Figure 2 and 4 inputs (≈40 min on a laptop)
python analysis/14_cohort_composition_h1model.py && python analysis/16_train109_predict25.py   # Table 3
python analysis/19_subgroup_difference_bootstrap.py                                            # contrasts, MDE
python analysis/18_da_m2_check.py                                                              # null-feature check
python analysis/20_table1_cleaning.py                                                          # Table 1
python analysis/17_figures_clinical.py                                                         # Figures 1–4
```

Seed 20260904 throughout. Two pre-registered deviations are recorded in Supplementary Note 1:
tree models run with five rather than ten repeats, and the sparse-group inner grid is 5 × 5
because `group_lasso` 1.5 is incompatible with scikit-learn 1.6's `GridSearchCV`.

## Rights

**No licence is granted for reuse.** This work was produced under grants from Taiwan's National
Science and Technology Council (NSTC 114-2221-E-039-006, 113-2221-E-039-006, 112-2221-E-039-006).
Under Article 2 of China Medical University's regulations on research results, intellectual
property in such results — including copyright — vests in the University, so the authors are not
in a position to grant a licence unilaterally.

The code is published here for one purpose: so that the analysis reported in the accompanying
manuscript can be inspected and verified. For any other use, please contact the corresponding
author (Yen-Fen Ko, kklven@gmail.com).

Chinese-language notes are in `README.zh-TW.md`.

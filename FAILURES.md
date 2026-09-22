# Failure modes, fixes, and results

Honest engineering notes for this project. Nothing here is invented for polish.

## What can go wrong

- **Country leakage** (same country in train and test). Impact: optimistic scores. Mitigation: country hold-out split as primary test.
- **Threshold tuned on the test set.** Impact: inflated F1. Mitigation: nested CV median threshold on training only (`THESIS_ALIGNMENT_AUDIT.md`).
- **Accuracy-led narrative on a ~5.5% default rate.** Impact: misleading thesis claims. Mitigation: primary metrics F1, recall, PR-AUC; thesis text patched.
- **Code/paper contradictions** (attention hybrid, SMOTE placement, Table 4.2 AUC band). Impact: unreproducible thesis. Mitigation: unified `src/` pipeline + alignment audit.

## What went wrong

Documented in `THESIS_ALIGNMENT_AUDIT.md` and commits (real research mistakes, then fixed):

1. Earlier notebooks/thesis claimed attention hybrid, nested/purged CV, and metrics that the code did not implement cleanly.
2. Threshold tuning on test inflated F1; Table 4.2 had contradictory AUCs (0.68-0.72 band); default rate stated as 1.4% vs ~5.5% in sample.
3. Hybrid model **F1 = 0** on the temporal split (nested CV threshold failure); country split remains the primary evidence base.

## How it was resolved

- Replaced repo with a single reproducible pipeline (`2126d49`), hybrid nested CV in `src/hybrid_tuning.py`, tests, and thesis regeneration scripts.
- `scripts/patch_thesis.py` / alignment commit removed stale claims and synced narrative to pipeline CSVs (`32526d8`).
- SMOTE confined to training; panel merge optional via World Bank build script.

## Results

Country hold-out (from audit table): Hybrid F1 **0.122**, Probit **0.121**, Logistic **0.110**, XGBoost **0.100**. Temporal: XGBoost best F1 **0.104**; Hybrid **0.000**. ROC AUC near 0.5 for most models (rare defaults). Regenerate with `python scripts/regenerate_thesis_results.py` and `pytest`.

# Thesis ↔ Code Alignment Audit

Generated after thesis aligned hybrid tuning (`src/hybrid_tuning.py`).

## Summary

| Area | Thesis claim (before) | Code evidence | Status |
|------|----------------------|---------------|--------|
| Attention LSTM | Attention based LSTM hybrid | `src/hybrid_tuning.py` + `src/models.py` | **Fixed** |
| Validation split | Nested / purged CV | Country hold out + temporal split + nested CV for hybrid | **Fixed** |
| Threshold tuning | On test set (inflated F1) | Nested CV median threshold on training only | **Fixed** |
| Hybrid stack | Attention context → XGBoost | Strict two stage stack (Section 3.5.2) | **Fixed** |
| SMOTE on hybrid | Both LSTM and XGBoost | SMOTE on sequences + embedding stage | **Fixed** |
| LSTM / XGB tuning | Random search + GridSearchCV | `nested_cv_select_hybrid_config()` | **Fixed** |
| Primary metrics | Accuracy led narrative | F1, recall, PR AUC first | **Fixed** |
| Table 4.2 | Contradictory AUC (0.68–0.72) | Single table from pipeline CSV | **Fixed** |
| Reproducibility | Scattered notebooks | `run_evaluation.py` + tests | **Fixed** |

## Country hold out results (primary test)

| Model | F1 | Recall | Precision | PR AUC | ROC AUC |
|-------|-----|--------|-----------|--------|---------|
| Hybrid (Attention LSTM + XGBoost) | **0.122** | 15.6% | **10.0%** | **0.081** | 0.509 |
| Probit | 0.121 | **65.7%** | 6.7% | 0.073 | **0.563** |
| Logistic | 0.110 | 100% | 5.8% | 0.072 | 0.559 |
| XGBoost | 0.100 | 57.1% | 5.5% | 0.059 | 0.494 |

## Temporal split results (robustness)

| Model | F1 | Recall | PR AUC | ROC AUC |
|-------|-----|--------|--------|---------|
| XGBoost | **0.104** | **76.9%** | **0.069** | **0.577** |
| Probit | 0.080 | 41.0% | 0.067 | 0.507 |
| Logistic | 0.078 | 28.2% | 0.067 | 0.502 |
| Hybrid | 0.000 | 0.0% | 0.039 | 0.418 |

## Remaining limitations (documented honestly)

1. **Hybrid temporal split** fails at nested CV threshold (F1 = 0); country split is the primary evidence base.
2. **Hybrid recall** on country split (15.6%) is far below Probit/logistic despite best F1.
3. **Probit joint significance** remains weak (pseudo R² ≈ 0.009; LLR p ≈ 0.73).
4. **ROC AUC** near 0.5 for most models reflects rare, heterogeneous defaults.
5. **Default rate** in full sample ≈ 5.5% (not 1.4%); thesis text patched to match.

## Thesis text patches (Jul 2026)

`scripts/patch_thesis.py` updates Table 4.2, abstract, hypotheses, methodology (nested CV, panel merge), Section 4.3–4.6, diagnostics (paired t test t = −0.33, p = 0.743), and limitations. Stale claims (ROC 0.68–0.72, 42% recall, 1.4% default rate, Hosmer–Lemeshow 100.71, 59% accuracy CI) removed.

## Regenerate everything

```bash
pip install -r requirements.txt
python scripts/regenerate_thesis_results.py
python -m pytest tests/test_pipeline.py
```

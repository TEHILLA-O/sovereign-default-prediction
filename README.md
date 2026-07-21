# Sovereign Default Prediction — Fixed Evaluation Pipeline

This folder contains a **single reproducible pipeline** that fixes the main issues in the earlier notebooks:

- **Country hold-out split** (no leakage across the same country)
- **Temporal split** (train on years ≤ 2010, test on later years)
- **Threshold tuning** on validation probabilities (maximize default-class F1)
- **SMOTE** on training data only (XGBoost / hybrid second stage)
- **Primary metrics**: F1, recall, PR-AUC (not accuracy)

## Dataset

`extras/sovereign_default_dataset_1980_2022.csv - sovereign_default_dataset_1980_2022.csv`

Optional enriched panel (World Bank overlay): `data_out/panel_final.csv`. When present, `load_dataset()` merges panel values onto the sovereign CSV by country and year.

Refresh the panel from World Bank:

```bash
pip install -r requirements-dev.txt
python scripts/build_panel.py
```

## Run

```bash
pip install -r requirements.txt
python scripts/regenerate_thesis_results.py
```

For metrics only (without full thesis figure/table refresh):

```bash
python run_evaluation.py
```

## Outputs

- `results/model_comparison_country_split.csv`
- `results/model_comparison_temporal_split.csv`
- `results/table_4_1_descriptive_statistics.csv`
- `results/evaluation_summary.json`
- `results/figures/figure_4_1_*.png` through `figure_5_2_*.png`
- Updated `extras/THESIS_OBANOR_TEHILLA_49_.docx` (tables, narrative, embedded figures)

## Source layout

- `src/panel_data.py` — optional panel_final merge and World Bank column normalization
- `scripts/build_panel.py` — build `data_out/panel_final.csv` from wbgapi
- `src/data.py` — loading and split helpers
- `src/metrics.py` — F1 / PR-AUC / threshold tuning
- `src/models.py` — Probit, Logistic, XGBoost, Hybrid (LSTM + XGBoost)
- `src/hybrid_tuning.py` — nested CV, LSTM random search, XGB grid, SMOTE sequences
- `src/thesis_results.py` — Table 4.1 and Figures 4.1–4.11, 5.1–5.2
- `src/thesis_text.py` — dynamic thesis paragraph and table builders
- `src/figure_export.py` — embed figures in the thesis docx
- `scripts/regenerate_thesis_results.py` — full thesis results regeneration

## Alignment audit

See [THESIS_ALIGNMENT_AUDIT.md](THESIS_ALIGNMENT_AUDIT.md) for thesis ↔ code mapping and corrected Chapter 4/5 narrative.

## Tests and CI

```bash
python -m pytest tests/ -q
```

GitHub Actions runs on push/PR to `main` (`.github/workflows/ci.yml`).

## Patch thesis document

After re-running evaluation:

```bash
python scripts/patch_thesis.py
```

Creates a timestamped backup of the `.docx` in `extras/` before applying text updates.

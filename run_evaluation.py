from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR, SELECTED_FEATURES
from src.data import (
    load_dataset,
    sequence_split,
    split_countries,
    tabular_split,
    temporal_split,
)
from src.models import (
    fit_hybrid_lag_xgb,
    fit_hybrid_lstm_xgb,
    fit_logistic,
    fit_probit,
    fit_xgboost,
)


def _metrics_row(name: str, result: dict) -> dict:
    m = result["metrics"]
    row = {
        "model": name,
        "threshold": round(result.get("threshold", m["threshold"]), 3),
        "f1_default": round(m["f1_default"], 4),
        "recall_default": round(m["recall_default"], 4),
        "precision_default": round(m["precision_default"], 4),
        "pr_auc": round(m["pr_auc"], 4) if m["pr_auc"] == m["pr_auc"] else None,
        "roc_auc": round(m["roc_auc"], 4) if m["roc_auc"] == m["roc_auc"] else None,
        "accuracy": round(m["accuracy"], 4),
        "test_defaults": m["support_default"],
        "test_observations": m["support_total"],
    }

    if "pseudo_r2" in result:
        row["pseudo_r2"] = round(result["pseudo_r2"], 4)
        row["llr_pvalue"] = round(result["llr_pvalue"], 4)

    return row


def _train_hybrid(train_df, test_df, train_countries, test_countries, df):
    X_train_seq, X_test_seq, y_train_seq, y_test_seq, _ = sequence_split(
        df, SELECTED_FEATURES, train_countries, test_countries
    )

    try:
        import tensorflow

        return fit_hybrid_lstm_xgb(X_train_seq, y_train_seq, X_test_seq, y_test_seq)
    except ImportError:
        print("TensorFlow not available; using lag-feature hybrid fallback.")
        return fit_hybrid_lag_xgb(train_df, test_df, SELECTED_FEATURES)


def _train_hybrid_temporal(train_df, test_df):
    from src.data import build_sequences

    X_train_seq, y_train_seq, _ = build_sequences(train_df, SELECTED_FEATURES)
    X_test_seq, y_test_seq, _ = build_sequences(test_df, SELECTED_FEATURES)

    try:
        import tensorflow

        return fit_hybrid_lstm_xgb(X_train_seq, y_train_seq, X_test_seq, y_test_seq)
    except ImportError:
        return fit_hybrid_lag_xgb(train_df, test_df, SELECTED_FEATURES)


def evaluate_split(name: str, X_train, X_test, y_train, y_test, train_df, test_df, df=None, split_meta=None):
    print(f"\n=== {name} ===")
    if split_meta:
        print(f"Train countries: {len(split_meta[0])} | Test countries: {len(split_meta[1])}")

    results = []

    probit = fit_probit(X_train, y_train, X_test, y_test, SELECTED_FEATURES)
    logistic = fit_logistic(X_train, y_train, X_test, y_test)
    xgb = fit_xgboost(X_train, y_train, X_test, y_test, use_smote=True)

    results.extend(
        [
            _metrics_row("Probit", probit),
            _metrics_row("Logistic Regression", logistic),
            _metrics_row("XGBoost", xgb),
        ]
    )

    if df is not None and split_meta and "country" in name.lower():
        train_countries, test_countries = split_meta
        hybrid = _train_hybrid(train_df, test_df, train_countries, test_countries, df)
    else:
        hybrid = _train_hybrid_temporal(train_df, test_df)

    results.append(_metrics_row(hybrid["model"], hybrid))

    results_df = pd.DataFrame(results).sort_values("f1_default", ascending=False)
    display_cols = [
        "model",
        "f1_default",
        "recall_default",
        "precision_default",
        "pr_auc",
        "roc_auc",
        "accuracy",
        "threshold",
    ]
    print(results_df[display_cols].to_string(index=False))

    return results_df, {
        "probit": probit,
        "logistic": logistic,
        "xgboost": xgb,
        "hybrid": hybrid,
    }


def run_evaluation(regenerate_thesis: bool = False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    train_countries, test_countries = split_countries(df)

    X_train, X_test, y_train, y_test, train_df, test_df = tabular_split(
        df, SELECTED_FEATURES, train_countries, test_countries
    )

    country_results, country_detail = evaluate_split(
        "Country hold-out split",
        X_train,
        X_test,
        y_train,
        y_test,
        train_df,
        test_df,
        df=df,
        split_meta=(train_countries, test_countries),
    )

    X_train_t, X_test_t, y_train_t, y_test_t, train_df_t, test_df_t = temporal_split(
        df, SELECTED_FEATURES, cutoff_year=2010
    )
    temporal_results, temporal_detail = evaluate_split(
        "Temporal split (train <= 2010, test > 2010)",
        X_train_t,
        X_test_t,
        y_train_t,
        y_test_t,
        train_df_t,
        test_df_t,
    )

    country_path = RESULTS_DIR / "model_comparison_country_split.csv"
    temporal_path = RESULTS_DIR / "model_comparison_temporal_split.csv"
    country_results.to_csv(country_path, index=False)
    temporal_results.to_csv(temporal_path, index=False)

    summary = {
        "dataset": str(Path("extras/sovereign_default_dataset_1980_2022.csv - sovereign_default_dataset_1980_2022.csv")),
        "primary_metrics": ["f1_default", "recall_default", "pr_auc"],
        "notes": [
            "Hybrid: SMOTE on LSTM sequences and XGBoost embeddings, nested CV for LSTM random search and XGB grid.",
            "Hybrid stack is attention context embeddings to XGBoost only (Section 3.5.2).",
            "Classification thresholds for all models come from inner validation or nested CV on training data only.",
            "SMOTE applied on training data only for tree models.",
            "Country split prevents the same country appearing in train and test.",
        ],
        "country_split": {
            "train_countries": sorted(train_countries),
            "test_countries": sorted(test_countries),
            "probit_pseudo_r2": country_detail["probit"]["pseudo_r2"],
            "probit_llr_pvalue": country_detail["probit"]["llr_pvalue"],
        },
        "country_results": country_results.to_dict(orient="records"),
        "temporal_results": temporal_results.to_dict(orient="records"),
    }

    json_path = RESULTS_DIR / "evaluation_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nSaved:\n- {country_path}\n- {temporal_path}\n- {json_path}")

    if not regenerate_thesis:
        from src.figure_export import export_thesis_roc_figures, save_roc_plot_data

        save_roc_plot_data(country_detail["xgboost"], country_detail["hybrid"])
        export_thesis_roc_figures(
            xgb_result=country_detail["xgboost"],
            hybrid_result=country_detail["hybrid"],
        )

    return country_results, temporal_results, country_detail, df, X_train, y_train, X_test


if __name__ == "__main__":
    run_evaluation()

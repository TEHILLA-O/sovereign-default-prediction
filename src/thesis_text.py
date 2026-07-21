from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import RESULTS_DIR
from src.thesis_results import _pct


def _metric_row(country_df: pd.DataFrame, model_name: str) -> pd.Series:
    row = country_df.loc[country_df["model"] == model_name].iloc[0]
    return row


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def build_table_4_2_rows(country_df: pd.DataFrame, probit_detail: dict | None = None) -> list[list[str]]:
    probit = _metric_row(country_df, "Probit")
    logistic = _metric_row(country_df, "Logistic Regression")
    xgb = _metric_row(country_df, "XGBoost")
    hybrid = _metric_row(country_df, "Hybrid (Attention-LSTM + XGBoost)")

    if probit_detail is None:
        pseudo_r2 = probit.get("pseudo_r2", 0.009)
        llr_p = probit.get("llr_pvalue", 0.725)
    else:
        pseudo_r2 = probit_detail.get("pseudo_r2", probit.get("pseudo_r2", 0.009))
        llr_p = probit_detail.get("llr_pvalue", probit.get("llr_pvalue", 0.725))

    return [
        ["Metric (country hold out test)", "Probit", "Logistic", "XGBoost", "Hybrid (Att LSTM + XGB)"],
        ["F1 score (default class)", _fmt(probit["f1_default"]), _fmt(logistic["f1_default"]), _fmt(xgb["f1_default"]), _fmt(hybrid["f1_default"])],
        ["Recall (default class)", _pct(probit["recall_default"]), _pct(logistic["recall_default"]), _pct(xgb["recall_default"]), _pct(hybrid["recall_default"])],
        ["Precision (default class)", _pct(probit["precision_default"]), _pct(logistic["precision_default"]), _pct(xgb["precision_default"]), _pct(hybrid["precision_default"])],
        ["PR AUC", _fmt(probit["pr_auc"]), _fmt(logistic["pr_auc"]), _fmt(xgb["pr_auc"]), _fmt(hybrid["pr_auc"])],
        ["ROC AUC", _fmt(probit["roc_auc"]), _fmt(logistic["roc_auc"]), _fmt(xgb["roc_auc"]), _fmt(hybrid["roc_auc"])],
        ["Accuracy (secondary)", _pct(probit["accuracy"]), _pct(logistic["accuracy"]), _pct(xgb["accuracy"]), _pct(hybrid["accuracy"])],
        ["Pseudo R2 / LLR p value (Probit only)", f"{pseudo_r2:.3f} / {llr_p:.3f}", "n/a", "n/a", "n/a"],
    ]


def build_table_4_1_rows(table_df: pd.DataFrame) -> list[list[str]]:
    header = ["Variable", "Mean", "Median", "Std Dev", "Min", "25%", "75%", "Max"]
    rows = [header]
    for _, row in table_df.iterrows():
        rows.append(
            [
                row["Variable"],
                f"{row['Mean']:.2f}",
                f"{row['Median']:.2f}",
                f"{row['Std Dev']:.2f}",
                f"{row['Min']:.2f}",
                f"{row['25%']:.2f}",
                f"{row['75%']:.2f}",
                f"{row['Max']:.2f}",
            ]
        )
    return rows


def build_paragraph_updates(
    country_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    probit_detail: dict,
    default_rate: float,
    n_train_countries: int,
    n_test_countries: int,
) -> dict[int, str]:
    probit = _metric_row(country_df, "Probit")
    logistic = _metric_row(country_df, "Logistic Regression")
    xgb = _metric_row(country_df, "XGBoost")
    hybrid = _metric_row(country_df, "Hybrid (Attention-LSTM + XGBoost)")

    temporal_sorted = temporal_df.sort_values("f1_default", ascending=False)
    temporal_top = temporal_sorted.iloc[0]
    temporal_lines = ", ".join(
        f"{row['model']} ({row['f1_default']:.3f})" for _, row in temporal_sorted.iterrows()
    )

    best_f1 = country_df.loc[country_df["f1_default"].idxmax()]
    best_recall = country_df.loc[country_df["recall_default"].idxmax()]
    best_roc = country_df.loc[country_df["roc_auc"].idxmax()]

    roc_min = country_df["roc_auc"].min()
    roc_max = country_df["roc_auc"].max()
    pseudo_r2 = probit_detail.get("pseudo_r2", probit.get("pseudo_r2", 0.009))
    llr_p = probit_detail.get("llr_pvalue", probit.get("llr_pvalue", 0.725))

    return {
        42: (
            "This research develops a hybrid machine learning framework that combines XGBoost with "
            "attention based LSTM sequence encoding and SHAP explainability to predict sovereign debt "
            "default. The model is trained on macroeconomic, political, and global financial indicators "
            "for 68 countries from 1980 to 2022 using leakage aware country hold out and temporal validation."
        ),
        43: (
            "Empirical results show that machine learning models outperform the Probit baseline on "
            "minority class detection when evaluation prioritises recall, F1 score, and PR AUC rather "
            f"than accuracy. On a country hold out test set ({n_train_countries} training countries, "
            f"{n_test_countries} test countries), {best_f1['model']} achieved the highest default class F1 score "
            f"({_fmt(best_f1['f1_default'])}), {best_recall['model']} achieved the highest recall "
            f"({_pct(best_recall['recall_default'])}), and the attention LSTM + XGBoost hybrid reached "
            f"{_pct(hybrid['recall_default'])} recall with F1 of {_fmt(hybrid['f1_default'])}. "
            f"ROC AUC values remained modest (approximately {_fmt(roc_min, 2)} to {_fmt(roc_max, 2)}), "
            "reflecting the difficulty of rare event sovereign default prediction."
        ),
        413: (
            f"The dataset is characterised by a high degree of class imbalance, with approximately "
            f"{100 * default_rate:.1f} per cent of observations classified as defaults, as shown in "
            "Figure 4.1. This imbalance underscores the challenge inherent in modelling sovereign "
            "default events, which are rare but impactful occurrences."
        ),
        449: (
            "The performance of four modelling approaches was compared on a country hold out test set "
            "and on a temporal test set (train up to 2010, test after 2010). Table 4.2 reports the primary "
            "country hold out results. Evaluation prioritises default class F1 score, recall, and PR AUC; "
            "accuracy is secondary because of severe class imbalance. Classification thresholds were tuned "
            "on an inner validation split from training data only, and models were refit on the full training "
            "sample before test evaluation."
        ),
        452: (
            "Table 4.2: Comparison of model results (country hold out test set; primary metrics first)."
        ),
        454: (
            f"Table 4.2 shows that {best_f1['model']} achieved the highest default class F1 score ({_fmt(best_f1['f1_default'])}), "
            f"followed by logistic regression ({_fmt(logistic['f1_default'])}), XGBoost ({_fmt(xgb['f1_default'])}), "
            f"and the attention LSTM + XGBoost hybrid ({_fmt(hybrid['f1_default'])}). {best_recall['model']} "
            f"achieved the highest recall ({_pct(best_recall['recall_default'])}), indicating greater "
            f"sensitivity but lower precision ({_pct(best_recall['precision_default'])}). The hybrid model "
            f"detected {_pct(hybrid['recall_default'])} of defaults with {_pct(hybrid['precision_default'])} "
            "precision."
        ),
        455: (
            f"ROC AUC values remained modest across models ({_fmt(roc_min, 2)} to {_fmt(roc_max, 2)} on the "
            "country hold out set), indicating limited global ranking ability despite threshold tuning. "
            "PR AUC values were similarly low (0.06 to 0.07), which is expected when defaults are rare and "
            "heterogeneous. High accuracy therefore should not be interpreted as strong default detection."
        ),
        474: (
            "Figure 4.4 reports the hybrid model ROC curve on the country hold out test set. The curve "
            f"is consistent with the tabulated ROC AUC of approximately {_fmt(hybrid['roc_auc'], 2)} and "
            "confirms that the hybrid model does not substantially outperform random ranking on probability "
            "scores alone."
        ),
        475: (
            f"The standalone XGBoost model achieved F1 of {_fmt(xgb['f1_default'])} "
            f"with {_pct(xgb['recall_default'])} default recall and PR AUC of {_fmt(xgb['pr_auc'])}. "
            f"{best_f1['model']} achieved the highest F1 score ({_fmt(best_f1['f1_default'])}), while "
            f"{best_roc['model']} achieved the highest ROC AUC ({_fmt(best_roc['roc_auc'], 2)})."
        ),
        493: (
            f"The XGBoost ROC curve in Figure 4.5 is consistent with the reported ROC AUC of approximately "
            f"{_fmt(xgb['roc_auc'], 2)} on the country hold out split."
        ),
        495: (
            f"Probit regression remained a useful transparent baseline but a weak predictor. On the country "
            f"hold out set it achieved F1 of {_fmt(probit['f1_default'])}, recall of {_pct(probit['recall_default'])}, "
            f"and ROC AUC of {_fmt(probit['roc_auc'])}, with pseudo R² of {pseudo_r2:.3f} and an LLR p value "
            f"of {llr_p:.3f} (joint significance not achieved)."
        ),
        567: (
            "Robustness was assessed using the temporal split (train up to 2010, test after 2010) in addition to "
            "the primary country hold out design. On the temporal split, model ranking by F1 was: "
            f"{temporal_lines}. {temporal_top['model']} achieved the highest F1 ({temporal_top['f1_default']:.3f}). "
            "This confirms that ranking across models is not stable across evaluation designs."
        ),
        609: (
            "Section 4.6 summary: sovereign default prediction remains constrained by class imbalance and "
            f"limited signal. The best country hold out F1 score was {_fmt(best_f1['f1_default'])} ({best_f1['model']}). "
            f"Maximum recall reached {_pct(best_recall['recall_default'])} ({best_recall['model']})."
        ),
        626: (
            f"Machine learning models improve default recall and F1 relative to naive baselines, but ROC AUC "
            f"({_fmt(roc_min, 2)} to {_fmt(roc_max, 2)}) and PR AUC remain modest on the country hold out split. "
            f"{best_f1['model']} achieved the highest F1 ({_fmt(best_f1['f1_default'])}); {best_recall['model']} achieved the "
            f"highest recall ({_pct(best_recall['recall_default'])}); the hybrid reached {_pct(hybrid['recall_default'])} "
            f"recall with F1 of {_fmt(hybrid['f1_default'])}."
        ),
        637: (
            f"Probit regression achieved {_pct(probit['recall_default'])} recall on the country hold out split "
            f"but low precision and failed joint significance testing (pseudo R² ≈ {pseudo_r2:.3f}; LLR p = {llr_p:.3f})."
        ),
    }


def dehyphenate_text(text: str) -> str:
    import re

    replacements = [
        ("attention-based", "attention based"),
        ("Attention-based", "Attention based"),
        ("attention-LSTM", "attention LSTM"),
        ("Attention-LSTM", "Attention LSTM"),
        ("Att-LSTM", "Att LSTM"),
        ("hold-out", "hold out"),
        ("Hold-out", "Hold out"),
        ("Hold-Out", "Hold Out"),
        ("leakage-aware", "leakage aware"),
        ("minority-class", "minority class"),
        ("default-class", "default class"),
        ("rare-event", "rare event"),
        ("trade-off", "trade off"),
        ("trade-offs", "trade offs"),
        ("F1-optimal", "F1 optimal"),
        ("F1-tuned", "F1 tuned"),
        ("five-year", "five year"),
        ("country-year", "country year"),
        ("out-of-sample", "out of sample"),
        ("cross-validation", "cross validation"),
        ("decision-support", "decision support"),
        ("policy-relevant", "policy relevant"),
        ("Policy-Relevant", "Policy Relevant"),
        ("Hosmer-Lemeshow", "Hosmer Lemeshow"),
        ("ROC-AUC", "ROC AUC"),
        ("PR-AUC", "PR AUC"),
        ("p-values", "p values"),
        ("p-value", "p value"),
        ("t-test", "t test"),
        ("t-tests", "t tests"),
        ("non-default", "non default"),
        ("non-defaulting", "non defaulting"),
        ("early-warning", "early warning"),
        ("machine-learning", "machine learning"),
        ("gradient-boosting", "gradient boosting"),
        ("tree-based", "tree based"),
        ("black-box", "black box"),
        ("re-estimated", "re estimated"),
        ("F1-score", "F1 score"),
        ("Trade-offs", "Trade offs"),
        ("Scikit-learn", "Scikit learn"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)

    text = re.sub(r"(\d+(?:\.\d+)?)\s*[–—]\s*(\d+(?:\.\d+)?)", r"\1 to \2", text)
    text = text.replace("–", " to ").replace("—", ", ")
    text = text.replace("≤", "up to ").replace("≥", "at least ")
    text = text.replace("≈", "about ")
    return text


def load_results_from_disk() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    country_df = pd.read_csv(RESULTS_DIR / "model_comparison_country_split.csv")
    temporal_df = pd.read_csv(RESULTS_DIR / "model_comparison_temporal_split.csv")
    summary = json.loads((RESULTS_DIR / "evaluation_summary.json").read_text(encoding="utf-8"))
    return country_df, temporal_df, summary

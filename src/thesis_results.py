from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.metrics import precision_recall_curve, precision_score, recall_score

from src.config import ALL_FEATURES, RESULTS_DIR, SELECTED_FEATURES
from src.data import load_dataset, sequence_split, split_countries, tabular_split
from src.plots import plot_roc_curve

FIG_DIR = RESULTS_DIR / "figures"

TABLE_41_ROWS = [
    ("Debt_GDP", "Debt to GDP (%)"),
    ("ExtDebt_GDP", "External Debt to GDP (%)"),
    ("DebtServ_XGDP", "Debt Service to Exports (%)"),
    ("RealGDP_growth", "Real GDP Growth (%)"),
    ("GDP_per_capita_USD", "GDP per Capita (USD)"),
    ("Inflation", "Inflation (%)"),
    ("fiscal_balance", "Fiscal Balance (% GDP)"),
    ("primary_balance", "Primary Balance (% GDP)"),
    ("CurrentAccount_GDP", "Current Account Balance (% GDP)"),
    ("Reserves_months", "Reserves (Months of Imports)"),
    ("ExchangeRate_change", "Exchange Rate Change (%)"),
    ("Trade_openness", "Trade Openness (%)"),
    ("US_FedFundsRate", "US Fed Funds Rate (%)"),
    ("World_GDP_growth", "World GDP Growth (%)"),
    ("Oil_price_index", "Oil Price Index"),
    ("VIX", "VIX"),
    ("ICRG_political", "Political Stability (ICRG)"),
    ("ElectionYear", "Election Year (Dummy)"),
]

FIGURE_FILES = {
    "4.1": FIG_DIR / "figure_4_1_class_distribution.png",
    "4.2": FIG_DIR / "figure_4_2_feature_boxplots.png",
    "4.3": FIG_DIR / "figure_4_3_feature_importance.png",
    "4.4": FIG_DIR / "figure_4_4_hybrid_roc.png",
    "4.5": FIG_DIR / "figure_4_5_xgboost_roc.png",
    "4.6": FIG_DIR / "figure_4_6_precision_recall_thresholds.png",
    "4.7": FIG_DIR / "figure_4_7_shap_hybrid_beeswarm.png",
    "4.8": FIG_DIR / "figure_4_8_shap_hybrid_bar.png",
    "4.9": FIG_DIR / "figure_4_9_shap_xgb_beeswarm.png",
    "4.10": FIG_DIR / "figure_4_10_shap_xgb_bar.png",
    "4.11": FIG_DIR / "figure_4_11_paired_ttest.png",
    "5.1": FIG_DIR / "figure_5_1_linear_nonlinear.png",
    "5.2": FIG_DIR / "figure_5_2_probit_limitations.png",
}


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def build_table_4_1(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = load_dataset()
    rows = []
    for col, label in TABLE_41_ROWS:
        series = df[col].dropna()
        rows.append(
            {
                "Variable": label,
                "Mean": round(series.mean(), 2),
                "Median": round(series.median(), 2),
                "Std Dev": round(series.std(), 2),
                "Min": round(series.min(), 2),
                "25%": round(series.quantile(0.25), 2),
                "75%": round(series.quantile(0.75), 2),
                "Max": round(series.max(), 2),
            }
        )
    table = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / "table_4_1_descriptive_statistics.csv", index=False)
    return table


def plot_figure_4_1(df: pd.DataFrame, output_path: Path) -> Path:
    counts = df["Default"].value_counts().sort_index()
    labels = ["Non-default", "Default"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    total = sum(values)
    pcts = [100 * v / total for v in values]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=["#2ca02c", "#d62728"])
    ax.set_ylabel("Observations")
    ax.set_title("Figure 4.1: Class Distribution (1980 to 2022)")
    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height())}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_figure_4_2(df: pd.DataFrame, output_path: Path) -> Path:
    n_cols = 4
    n_rows = int(np.ceil(len(SELECTED_FEATURES) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
    axes = np.array(axes).reshape(-1)

    for idx, feature in enumerate(SELECTED_FEATURES):
        ax = axes[idx]
        subset = df[[feature, "Default"]].dropna()
        data = [subset.loc[subset["Default"] == 0, feature], subset.loc[subset["Default"] == 1, feature]]
        ax.boxplot(data, tick_labels=["No", "Yes"])
        ax.set_title(feature, fontsize=9)
        ax.set_xlabel("Default")

    for idx in range(len(SELECTED_FEATURES), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Figure 4.2: Feature Distributions by Default Status", y=1.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_figure_4_3(xgb_result: dict, output_path: Path) -> Path:
    model = xgb_result["model_obj"]
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    names = [SELECTED_FEATURES[i] for i in order]
    values = importances[order]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names[::-1], values[::-1], color="#1f77b4")
    ax.set_xlabel("Importance")
    ax.set_title("Figure 4.3: XGBoost Feature Importance Ranking")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_figure_4_6(hybrid_result: dict, xgb_result: dict, output_path: Path) -> Path:
    thresholds = np.arange(0.05, 0.96, 0.01)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, result, title in [
        (axes[0], hybrid_result, "Hybrid"),
        (axes[1], xgb_result, "XGBoost"),
    ]:
        y_true = result["y_test"]
        y_prob = result["y_prob"]
        precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
        ax.plot(thresholds, [precision_score(y_true, y_prob >= t, zero_division=0) for t in thresholds], label="Precision")
        ax.plot(thresholds, [recall_score(y_true, y_prob >= t, zero_division=0) for t in thresholds], label="Recall")
        ax.axvline(result["threshold"], color="red", linestyle="--", label=f"F1 threshold={result['threshold']:.2f}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Figure 4.6: Precision and Recall vs Threshold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_shap_plots(model, X_background, X_explain, feature_names, beeswarm_path: Path, bar_path: Path, title_prefix: str):
    predict_fn = lambda x: model.predict_proba(x)[:, 1]
    explainer = shap.Explainer(predict_fn, X_background)
    explanation = explainer(X_explain)
    shap_values = explanation.values
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_explain, feature_names=feature_names, show=False)
    plt.title(f"{title_prefix} Beeswarm")
    beeswarm_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_explain, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f"{title_prefix} Global Importance")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_shap_figures(
    country_detail: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[Path, Path, Path, Path]:
    xgb = country_detail["xgboost"]
    hybrid = country_detail["hybrid"]

    _save_shap_plots(
        xgb["model_obj"],
        X_train,
        X_test,
        SELECTED_FEATURES,
        FIGURE_FILES["4.9"],
        FIGURE_FILES["4.10"],
        "Figure 4.9/4.10: Standalone XGBoost SHAP",
    )

    embed_dim = hybrid["X_test_features"].shape[1]
    embed_names = [f"Embed_{i}" for i in range(embed_dim)]
    n_bg = min(200, hybrid["X_test_features"].shape[0])
    background = hybrid["X_test_features"][:n_bg]
    explain = hybrid["X_test_features"]

    _save_shap_plots(
        hybrid["model_obj"],
        background,
        explain,
        embed_names,
        FIGURE_FILES["4.7"],
        FIGURE_FILES["4.8"],
        "Figure 4.7/4.8: Hybrid SHAP",
    )
    return FIGURE_FILES["4.7"], FIGURE_FILES["4.8"], FIGURE_FILES["4.9"], FIGURE_FILES["4.10"]


def plot_figure_4_11(hybrid_result: dict, xgb_result: dict, output_path: Path) -> Path:
    hybrid_prob = hybrid_result["y_prob"]
    xgb_prob = xgb_result["y_prob"]
    n = min(len(hybrid_prob), len(xgb_prob))
    hybrid_prob = hybrid_prob[:n]
    xgb_prob = xgb_prob[:n]
    t_stat, p_value = stats.ttest_rel(hybrid_prob, xgb_prob)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(hybrid_prob, bins=30, alpha=0.6, label="Hybrid", color="#1f77b4")
    axes[0].hist(xgb_prob, bins=30, alpha=0.6, label="XGBoost", color="#ff7f0e")
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Probability distributions")
    axes[0].legend()

    axes[1].scatter(xgb_prob, hybrid_prob, alpha=0.4, s=12)
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("XGBoost probability")
    axes[1].set_ylabel("Hybrid probability")
    axes[1].set_title(f"Paired t test: t={t_stat:.2f}, p={p_value:.3f}")

    fig.suptitle("Figure 4.11: Paired Comparison of Model Probabilities")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(
        [{"t_statistic": t_stat, "p_value": p_value, "n_pairs": n}]
    ).to_csv(RESULTS_DIR / "figure_4_11_paired_ttest.csv", index=False)
    return output_path


def plot_figure_5_1(
    X_train: np.ndarray,
    y_train: np.ndarray,
    xgb_result: dict,
    output_path: Path,
) -> Path:
    import statsmodels.api as sm

    feat_a, feat_b = 2, 6
    x_a = X_train[:, feat_a]
    x_b = X_train[:, feat_b]
    grid_a = np.linspace(np.percentile(x_a, 5), np.percentile(x_a, 95), 80)
    grid_b = np.linspace(np.percentile(x_b, 5), np.percentile(x_b, 95), 80)
    aa, bb = np.meshgrid(grid_a, grid_b)

    X_grid = np.zeros((aa.size, X_train.shape[1]))
    X_grid[:, feat_a] = aa.ravel()
    X_grid[:, feat_b] = bb.ravel()
    X_grid_const = sm.add_constant(X_grid, has_constant="add")

    probit_model = sm.Probit(y_train, sm.add_constant(X_train, has_constant="add")).fit(disp=0)
    probit_prob = probit_model.predict(X_grid_const).reshape(aa.shape)
    xgb_prob = xgb_result["model_obj"].predict_proba(X_grid)[:, 1].reshape(aa.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, z, title in [
        (axes[0], probit_prob, "Probit (linear index)"),
        (axes[1], xgb_prob, "XGBoost (nonlinear)"),
    ]:
        contour = ax.contourf(aa, bb, z, levels=15, cmap="RdYlBu_r")
        fig.colorbar(contour, ax=ax, fraction=0.046)
        ax.scatter(x_a[y_train == 0], x_b[y_train == 0], s=8, c="green", alpha=0.25, label="No default")
        ax.scatter(x_a[y_train == 1], x_b[y_train == 1], s=12, c="red", alpha=0.5, label="Default")
        ax.set_xlabel(SELECTED_FEATURES[feat_a])
        ax.set_ylabel(SELECTED_FEATURES[feat_b])
        ax.set_title(title)
        ax.legend(fontsize=7)

    fig.suptitle("Figure 5.1: Linear vs Nonlinear Decision Surfaces")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_figure_5_2(probit_result: dict, output_path: Path) -> Path:
    y_true = probit_result["y_test"]
    y_prob = probit_result["y_prob"]
    pseudo_r2 = probit_result.get("pseudo_r2", np.nan)
    llr_p = probit_result.get("llr_pvalue", np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(y_prob[y_true == 0], bins=25, alpha=0.7, label="Non-default", color="#2ca02c")
    axes[0].hist(y_prob[y_true == 1], bins=25, alpha=0.7, label="Default", color="#d62728")
    axes[0].set_xlabel("Probit predicted probability")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Predicted probability overlap")
    axes[0].legend()

    axes[1].axis("off")
    axes[1].text(
        0.0,
        0.8,
        "Probit limitations (country hold-out test):\n\n"
        f"Pseudo R² = {pseudo_r2:.3f}\n"
        f"LLR p-value = {llr_p:.3f}\n"
        f"ROC-AUC = {probit_result['metrics']['roc_auc']:.3f}\n"
        f"Default recall = {_pct(probit_result['metrics']['recall_default'])}\n"
        f"Default precision = {_pct(probit_result['metrics']['precision_default'])}",
        fontsize=12,
        va="top",
    )

    fig.suptitle("Figure 5.2: Limitations of the Probit Baseline")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_all_thesis_results(
    country_detail: dict,
    df: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    build_table_4_1(df)

    paths = {}
    paths["4.1"] = plot_figure_4_1(df, FIGURE_FILES["4.1"])
    paths["4.2"] = plot_figure_4_2(df, FIGURE_FILES["4.2"])
    paths["4.3"] = plot_figure_4_3(country_detail["xgboost"], FIGURE_FILES["4.3"])

    hybrid = country_detail["hybrid"]
    xgb = country_detail["xgboost"]
    paths["4.4"] = plot_roc_curve(
        hybrid["y_test"],
        hybrid["y_prob"],
        "Figure 4.4: Hybrid Model ROC Curve (Country Hold Out Test)",
        FIGURE_FILES["4.4"],
        hybrid["threshold"],
        hybrid["metrics"]["roc_auc"],
    )
    paths["4.5"] = plot_roc_curve(
        xgb["y_test"],
        xgb["y_prob"],
        "Figure 4.5: XGBoost Model ROC Curve (Country Hold Out Test)",
        FIGURE_FILES["4.5"],
        xgb["threshold"],
        xgb["metrics"]["roc_auc"],
    )
    paths["4.6"] = plot_figure_4_6(hybrid, xgb, FIGURE_FILES["4.6"])
    plot_shap_figures(country_detail, X_train, X_test)
    paths["4.7"] = FIGURE_FILES["4.7"]
    paths["4.8"] = FIGURE_FILES["4.8"]
    paths["4.9"] = FIGURE_FILES["4.9"]
    paths["4.10"] = FIGURE_FILES["4.10"]
    paths["4.11"] = plot_figure_4_11(hybrid, xgb, FIGURE_FILES["4.11"])
    paths["5.1"] = plot_figure_5_1(X_train, y_train, xgb, FIGURE_FILES["5.1"])
    paths["5.2"] = plot_figure_5_2(country_detail["probit"], FIGURE_FILES["5.2"])

    for fig_id, path in paths.items():
        print(f"Saved Figure {fig_id}: {path}")
    return paths

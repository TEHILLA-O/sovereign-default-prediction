from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_evaluation import run_evaluation
from scripts.patch_thesis import patch_thesis_document
from src.figure_export import save_roc_plot_data, update_all_thesis_figures
from src.thesis_results import generate_all_thesis_results


def regenerate_thesis_results() -> None:
    country_results, temporal_results, country_detail, df, X_train, y_train, X_test = run_evaluation(
        regenerate_thesis=True
    )
    figure_paths = generate_all_thesis_results(country_detail, df, X_train, y_train, X_test)
    save_roc_plot_data(country_detail["xgboost"], country_detail["hybrid"])
    patch_thesis_document(country_df=country_results, temporal_df=temporal_results)
    update_all_thesis_figures(figure_paths)
    print("\nThesis results regeneration complete.")


if __name__ == "__main__":
    regenerate_thesis_results()

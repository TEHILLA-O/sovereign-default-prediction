from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from docx import Document

from src.config import PROJECT_ROOT, RESULTS_DIR, SELECTED_FEATURES
from src.data import load_dataset, sequence_split, split_countries, tabular_split
from src.models import fit_hybrid_lstm_xgb, fit_xgboost
from src.plots import plot_roc_curve

THESIS = PROJECT_ROOT / "extras" / "THESIS_OBANOR_TEHILLA_49_.docx"
FIG_DIR = RESULTS_DIR / "figures"
FIG_44 = FIG_DIR / "figure_4_4_hybrid_roc.png"
FIG_45 = FIG_DIR / "figure_4_5_xgboost_roc.png"
ROC_DATA_PATH = RESULTS_DIR / "roc_plot_data_country.npz"

FIGURE_MATCH_ORDER = ["4.10", "4.11", "4.9", "4.8", "4.7", "4.6", "4.5", "4.4", "4.3", "4.2", "4.1", "5.2", "5.1"]

BLIP_TAG = "{http://schemas.openxmlformats.org/drawingml/2006/main}blip"
REL_EMBED = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"


def _plot_inputs(result: dict) -> dict:
    return {
        "y_test": result["y_test"],
        "y_prob": result["y_prob"],
        "threshold": result["threshold"],
        "roc_auc": result["metrics"]["roc_auc"],
    }


def save_roc_plot_data(xgb_result: dict, hybrid_result: dict) -> Path:
    ROC_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        ROC_DATA_PATH,
        xgb_y_test=xgb_result["y_test"],
        xgb_y_prob=xgb_result["y_prob"],
        xgb_threshold=np.array([xgb_result["threshold"]]),
        xgb_roc_auc=np.array([xgb_result["metrics"]["roc_auc"]]),
        hybrid_y_test=hybrid_result["y_test"],
        hybrid_y_prob=hybrid_result["y_prob"],
        hybrid_threshold=np.array([hybrid_result["threshold"]]),
        hybrid_roc_auc=np.array([hybrid_result["metrics"]["roc_auc"]]),
    )
    return ROC_DATA_PATH


def _load_roc_plot_data() -> dict | None:
    if not ROC_DATA_PATH.exists():
        return None

    data = np.load(ROC_DATA_PATH, allow_pickle=True)
    return {
        "xgboost": {
            "y_test": data["xgb_y_test"],
            "y_prob": data["xgb_y_prob"],
            "threshold": float(data["xgb_threshold"][0]),
            "metrics": {"roc_auc": float(data["xgb_roc_auc"][0])},
        },
        "hybrid": {
            "y_test": data["hybrid_y_test"],
            "y_prob": data["hybrid_y_prob"],
            "threshold": float(data["hybrid_threshold"][0]),
            "metrics": {"roc_auc": float(data["hybrid_roc_auc"][0])},
        },
    }


def _train_country_models():
    df = load_dataset()
    train_countries, test_countries = split_countries(df)
    X_train, X_test, y_train, y_test, _, _ = tabular_split(
        df, SELECTED_FEATURES, train_countries, test_countries
    )
    xgb = fit_xgboost(X_train, y_train, X_test, y_test, use_smote=True)

    X_train_seq, X_test_seq, y_train_seq, y_test_seq, _ = sequence_split(
        df, SELECTED_FEATURES, train_countries, test_countries
    )
    hybrid = fit_hybrid_lstm_xgb(X_train_seq, y_train_seq, X_test_seq, y_test_seq)
    return xgb, hybrid


def generate_roc_figures(
    xgb_result: dict | None = None,
    hybrid_result: dict | None = None,
) -> tuple[Path, Path]:
    if xgb_result is None or hybrid_result is None:
        loaded = _load_roc_plot_data()
        if loaded:
            xgb_result = loaded["xgboost"]
            hybrid_result = loaded["hybrid"]
        else:
            xgb_result, hybrid_result = _train_country_models()

    hybrid = _plot_inputs(hybrid_result)
    xgb = _plot_inputs(xgb_result)

    plot_roc_curve(
        hybrid["y_test"],
        hybrid["y_prob"],
        "Figure 4.4: Hybrid Model ROC Curve (Country Hold-Out Test)",
        FIG_44,
        hybrid["threshold"],
        hybrid["roc_auc"],
    )
    plot_roc_curve(
        xgb["y_test"],
        xgb["y_prob"],
        "Figure 4.5: XGBoost Model ROC Curve (Country Hold-Out Test)",
        FIG_45,
        xgb["threshold"],
        xgb["roc_auc"],
    )

    print(f"Saved {FIG_44} (AUC={hybrid['roc_auc']:.4f})")
    print(f"Saved {FIG_45} (AUC={xgb['roc_auc']:.4f})")
    return FIG_44, FIG_45


def _shape_embed_id(shape) -> str:
    return shape._inline.graphic.graphicData.pic.blipFill.blip.embed


def _paragraph_has_embed(para, embed_id: str) -> bool:
    for run in para.runs:
        for blip in run._element.findall(f".//{BLIP_TAG}"):
            if blip.get(REL_EMBED) == embed_id:
                return True
    return False


def _caption_for_shape(doc: Document, shape_index: int) -> str:
    embed_id = _shape_embed_id(doc.inline_shapes[shape_index])
    caption = ""
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            caption = text
        if _paragraph_has_embed(para, embed_id):
            return caption
    return ""


def _image_captions(doc: Document) -> list[tuple[int, str]]:
    return [(idx, _caption_for_shape(doc, idx)) for idx in range(len(doc.inline_shapes))]


def _replace_inline_image(doc: Document, inline_shape, image_path: Path) -> None:
    r_id = inline_shape._inline.graphic.graphicData.pic.blipFill.blip.embed
    part = doc.part.related_parts[r_id]
    part._blob = image_path.read_bytes()


def _find_figure_indices(doc: Document) -> dict[str, list[int]]:
    found: dict[str, list[int]] = {fig_id: [] for fig_id in FIGURE_MATCH_ORDER}
    for idx, caption in _image_captions(doc):
        lower = caption.lower()
        for fig_id in FIGURE_MATCH_ORDER:
            if fig_id in lower:
                found[fig_id].append(idx)
                break
    return found


def update_thesis_figures(fig_44: Path, fig_45: Path) -> None:
    update_all_thesis_figures({"4.4": fig_44, "4.5": fig_45})


def update_all_thesis_figures(figure_paths: dict[str, Path]) -> None:
    backup = THESIS.with_name(
        f"THESIS_OBANOR_TEHILLA_49_FIGBACKUP_{datetime.now():%Y%m%d_%H%M%S}.docx"
    )
    shutil.copy2(THESIS, backup)
    doc = Document(THESIS)
    indices = _find_figure_indices(doc)

    missing = []
    replaced = 0
    for fig_id, path in figure_paths.items():
        if not path.exists():
            missing.append(f"{fig_id} ({path})")
            continue
        shape_indices = indices.get(fig_id, [])
        if not shape_indices:
            missing.append(fig_id)
            continue
        for shape_idx in shape_indices:
            _replace_inline_image(doc, doc.inline_shapes[shape_idx], path)
            replaced += 1

    doc.save(THESIS)
    print(f"Thesis figures updated: {replaced} images in {THESIS}")
    print(f"Backup: {backup}")
    if missing:
        print(f"Warning: could not replace figures: {missing}")


def export_thesis_roc_figures(
    xgb_result: dict | None = None,
    hybrid_result: dict | None = None,
) -> tuple[Path, Path]:
    fig_44, fig_45 = generate_roc_figures(xgb_result, hybrid_result)
    update_thesis_figures(fig_44, fig_45)
    return fig_44, fig_45

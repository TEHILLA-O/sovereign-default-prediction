from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SELECTED_FEATURES
from src.data import load_dataset, split_countries, tabular_split
from src.metrics import evaluate_predictions, tune_threshold
from src.models import fit_logistic, fit_xgboost
from src.validation import stratified_train_val_split


def test_country_holdout_has_no_overlap():
    df = load_dataset()
    train_countries, test_countries = split_countries(df)
    assert train_countries.isdisjoint(test_countries)


def test_validation_split_is_stratified():
    df = load_dataset()
    train_countries, test_countries = split_countries(df)
    X_train, _, y_train, _, _, _ = tabular_split(
        df, SELECTED_FEATURES, train_countries, test_countries
    )
    _, _, y_tr, y_val = stratified_train_val_split(X_train, y_train)
    assert len(y_tr) + len(y_val) == len(y_train)
    assert y_tr.sum() >= 1
    assert y_val.sum() >= 1


def test_tune_threshold_maximizes_f1_on_well_separated_probs():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    threshold, f1 = tune_threshold(y_true, y_prob, metric="f1")
    y_pred = (y_prob >= threshold).astype(int)

    assert 0.4 <= threshold <= 0.6
    assert f1 == 1.0
    assert y_pred.sum() == y_true.sum()


def test_evaluate_predictions_returns_expected_keys():
    y_true = np.array([0, 0, 1, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.7, 0.3])
    result = evaluate_predictions(y_true, y_prob, threshold=0.5)
    for key in ["f1_default", "recall_default", "precision_default", "roc_auc", "pr_auc"]:
        assert key in result


def test_xgboost_uses_validation_thresholding():
    df = load_dataset()
    train_countries, test_countries = split_countries(df)
    X_train, X_test, y_train, y_test, _, _ = tabular_split(
        df, SELECTED_FEATURES, train_countries, test_countries
    )
    result = fit_xgboost(X_train, y_train, X_test, y_test)
    assert 0.0 < result["threshold"] < 1.0
    assert result["metrics"]["f1_default"] >= 0.0


def test_logistic_runs_end_to_end():
    df = load_dataset()
    train_countries, test_countries = split_countries(df)
    X_train, X_test, y_train, y_test, _, _ = tabular_split(
        df, SELECTED_FEATURES, train_countries, test_countries
    )
    result = fit_logistic(X_train, y_train, X_test, y_test)
    assert len(result["y_prob"]) == len(y_test)


def test_hybrid_thesis_pipeline_fast_mode():
    pytest.importorskip("tensorflow")
    from src.data import sequence_split
    from src.models import fit_hybrid_lstm_xgb

    df = load_dataset()
    train_countries, test_countries = split_countries(df)
    X_train_seq, X_test_seq, y_train_seq, y_test_seq, _ = sequence_split(
        df, SELECTED_FEATURES, train_countries, test_countries
    )
    result = fit_hybrid_lstm_xgb(
        X_train_seq,
        y_train_seq,
        X_test_seq,
        y_test_seq,
        fast_mode=True,
    )
    assert result["X_test_features"].shape[0] == len(y_test_seq)
    assert result["X_test_features"].ndim == 2
    assert 0.0 < result["threshold"] < 1.0
    assert "lstm_params" in result
    assert "xgb_params" in result

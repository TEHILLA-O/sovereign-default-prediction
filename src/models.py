from __future__ import annotations

import warnings

import numpy as np
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import HYBRID_XGB_PARAMS, RANDOM_STATE, XGB_PARAMS
from .metrics import evaluate_predictions, tune_threshold
from .validation import stratified_train_val_split

warnings.filterwarnings("ignore", category=FutureWarning)


def _set_global_seeds():
    import os
    import random

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
    try:
        import tensorflow as tf

        tf.random.set_seed(RANDOM_STATE)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except ImportError:
        pass


def _tune_threshold_on_validation(y_val, y_prob_val):
    return tune_threshold(y_val, y_prob_val, metric="f1")


def _smote_resample(X, y):
    if y.sum() >= 2:
        smote = SMOTE(random_state=RANDOM_STATE)
        return smote.fit_resample(X, y)
    return X, y


def _fit_xgb_model(X_train, y_train, X_eval, y_eval, params, early_stopping_rounds=25):
    X_res, y_res = _smote_resample(X_train, y_train)
    scale_pos_weight = max(1.0, (len(y_res) - y_res.sum()) / max(y_res.sum(), 1))

    model = XGBClassifier(
        **{k: v for k, v in params.items() if k not in {"use_label_encoder", "early_stopping_rounds"}},
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=early_stopping_rounds,
    )
    model.fit(X_res, y_res, eval_set=[(X_eval, y_eval)], verbose=False)
    return model


def _refit_xgb_model(X_train, y_train, params, n_estimators):
    X_res, y_res = _smote_resample(X_train, y_train)
    scale_pos_weight = max(1.0, (len(y_res) - y_res.sum()) / max(y_res.sum(), 1))
    fit_params = {
        **{k: v for k, v in params.items() if k not in {"use_label_encoder", "early_stopping_rounds", "n_estimators"}},
        "n_estimators": max(50, int(n_estimators)),
        "scale_pos_weight": scale_pos_weight,
    }
    model = XGBClassifier(**fit_params)
    model.fit(X_res, y_res, verbose=False)
    return model


def fit_probit(X_train, y_train, X_test, y_test, feature_names):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_tr, X_val, y_tr, y_val = stratified_train_val_split(X_train_scaled, y_train)

    X_tr_const = sm.add_constant(X_tr, has_constant="add")
    X_val_const = sm.add_constant(X_val, has_constant="add")
    X_test_const = sm.add_constant(X_test_scaled, has_constant="add")
    X_full_const = sm.add_constant(X_train_scaled, has_constant="add")

    dev_model = sm.Probit(y_tr, X_tr_const).fit(disp=0, maxiter=250)
    y_prob_val = dev_model.predict(X_val_const)
    threshold, _ = _tune_threshold_on_validation(y_val, y_prob_val)

    full_model = sm.Probit(y_train, X_full_const).fit(disp=0, maxiter=250)
    y_prob_test = full_model.predict(X_test_const)
    metrics = evaluate_predictions(y_test, y_prob_test, threshold)

    pseudo_r2 = float(getattr(full_model, "prsquared", np.nan))
    llr_pvalue = float(full_model.llr_pvalue) if full_model.llr_pvalue is not None else np.nan

    return {
        "model": "Probit",
        "metrics": metrics,
        "threshold": threshold,
        "pseudo_r2": pseudo_r2,
        "llr_pvalue": llr_pvalue,
        "y_test": y_test,
        "y_prob": y_prob_test,
        "summary": {
            "n_train": len(y_train),
            "n_test": len(y_test),
            "default_rate_test": float(y_test.mean()),
        },
    }


def fit_logistic(X_train, y_train, X_test, y_test):
    X_tr, X_val, y_tr, y_val = stratified_train_val_split(X_train, y_train)

    dev_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                    C=1.0,
                ),
            ),
        ]
    )
    dev_pipeline.fit(X_tr, y_tr)
    y_prob_val = dev_pipeline.predict_proba(X_val)[:, 1]
    threshold, _ = _tune_threshold_on_validation(y_val, y_prob_val)

    final_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                    C=1.0,
                ),
            ),
        ]
    )
    final_pipeline.fit(X_train, y_train)
    y_prob_test = final_pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(y_test, y_prob_test, threshold)

    return {
        "model": "Logistic Regression",
        "metrics": metrics,
        "threshold": threshold,
        "pipeline": final_pipeline,
        "y_test": y_test,
        "y_prob": y_prob_test,
    }


def fit_xgboost(X_train, y_train, X_test, y_test, use_smote: bool = True):
    X_tr, X_val, y_tr, y_val = stratified_train_val_split(X_train, y_train)

    if not use_smote:
        dev_model = XGBClassifier(
            **{k: v for k, v in XGB_PARAMS.items() if k != "use_label_encoder"},
            scale_pos_weight=max(1.0, (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)),
            early_stopping_rounds=XGB_PARAMS.get("early_stopping_rounds", 25),
        )
        dev_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    else:
        dev_model = _fit_xgb_model(
            X_tr,
            y_tr,
            X_val,
            y_val,
            XGB_PARAMS,
            early_stopping_rounds=XGB_PARAMS.get("early_stopping_rounds", 25),
        )

    y_prob_val = dev_model.predict_proba(X_val)[:, 1]
    threshold, _ = _tune_threshold_on_validation(y_val, y_prob_val)

    best_iteration = getattr(dev_model, "best_iteration", None)
    if best_iteration is None or best_iteration <= 0:
        best_iteration = XGB_PARAMS.get("n_estimators", 400)

    final_model = _refit_xgb_model(X_train, y_train, XGB_PARAMS, best_iteration + 1)
    y_prob_test = final_model.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(y_test, y_prob_test, threshold)

    return {
        "model": "XGBoost",
        "metrics": metrics,
        "threshold": threshold,
        "model_obj": final_model,
        "y_test": y_test,
        "y_prob": y_prob_test,
    }


def _sequence_tabular_features(X_seq):
    last_step = X_seq[:, -1, :]
    mean_step = X_seq.mean(axis=1)
    return np.hstack([last_step, mean_step])


def fit_hybrid_lstm_xgb(X_train_seq, y_train, X_test_seq, y_test, fast_mode: bool = False):
    from .config import HYBRID_LSTM_RANDOM_TRIALS, HYBRID_OUTER_FOLDS, HYBRID_XGB_PARAM_GRID
    from .hybrid_tuning import (
        extract_attention_context,
        fit_hybrid_xgb,
        fit_lstm_encoder,
        nested_cv_select_hybrid_config,
        refit_hybrid_xgb,
    )

    _set_global_seeds()

    outer_folds = 2 if fast_mode else HYBRID_OUTER_FOLDS
    lstm_trials = 2 if fast_mode else HYBRID_LSTM_RANDOM_TRIALS
    xgb_grid = HYBRID_XGB_PARAM_GRID[:2] if fast_mode else HYBRID_XGB_PARAM_GRID

    selected = nested_cv_select_hybrid_config(
        X_train_seq,
        y_train,
        outer_folds=outer_folds,
        lstm_trials=lstm_trials,
        xgb_grid=xgb_grid,
    )

    X_tr, X_val, y_tr, y_val = stratified_train_val_split(X_train_seq, y_train)
    final_lstm = fit_lstm_encoder(
        X_tr,
        y_tr,
        X_val,
        y_val,
        selected["lstm_params"],
        use_smote=True,
    )

    embed_train = extract_attention_context(final_lstm, X_train_seq)
    embed_test = extract_attention_context(final_lstm, X_test_seq)
    embed_tr = extract_attention_context(final_lstm, X_tr)
    embed_val = extract_attention_context(final_lstm, X_val)

    dev_xgb = fit_hybrid_xgb(
        embed_tr,
        y_tr,
        embed_val,
        y_val,
        selected["xgb_params"],
    )

    best_iteration = getattr(dev_xgb, "best_iteration", None)
    if best_iteration is None or best_iteration <= 0:
        best_iteration = selected["xgb_params"]["n_estimators"]

    final_xgb = refit_hybrid_xgb(
        embed_train,
        y_train,
        selected["xgb_params"],
        best_iteration + 1,
    )

    threshold = selected["threshold"]
    y_prob_test = final_xgb.predict_proba(embed_test)[:, 1]
    metrics = evaluate_predictions(y_test, y_prob_test, threshold)

    return {
        "model": "Hybrid (Attention-LSTM + XGBoost)",
        "metrics": metrics,
        "threshold": threshold,
        "model_obj": final_xgb,
        "lstm_params": selected["lstm_params"],
        "xgb_params": selected["xgb_params"],
        "nested_cv_f1": selected["mean_f1"],
        "y_test": y_test,
        "y_prob": y_prob_test,
        "X_test_features": embed_test,
    }


def fit_hybrid_lag_xgb(train_df, test_df, features, sequence_length=5):
    from .data import build_sequences

    train_seq, train_y, _ = build_sequences(train_df, features, sequence_length)
    test_seq, test_y, _ = build_sequences(test_df, features, sequence_length)

    X_train_flat = np.hstack([train_seq.reshape(train_seq.shape[0], -1), _sequence_tabular_features(train_seq)])
    X_test_flat = np.hstack([test_seq.reshape(test_seq.shape[0], -1), _sequence_tabular_features(test_seq)])

    X_tr, X_val, y_tr, y_val = stratified_train_val_split(X_train_flat, train_y)

    dev_xgb = _fit_xgb_model(
        X_tr,
        y_tr,
        X_val,
        y_val,
        HYBRID_XGB_PARAMS,
        early_stopping_rounds=HYBRID_XGB_PARAMS.get("early_stopping_rounds", 25),
    )
    y_prob_val = dev_xgb.predict_proba(X_val)[:, 1]
    threshold, _ = _tune_threshold_on_validation(y_val, y_prob_val)

    best_iteration = getattr(dev_xgb, "best_iteration", HYBRID_XGB_PARAMS.get("n_estimators", 500))
    final_xgb = _refit_xgb_model(
        X_train_flat,
        train_y,
        HYBRID_XGB_PARAMS,
        best_iteration + 1,
    )
    y_prob_test = final_xgb.predict_proba(X_test_flat)[:, 1]
    metrics = evaluate_predictions(y_test, y_prob_test, threshold)

    return {
        "model": "Hybrid (Attention-LSTM + XGBoost)",
        "metrics": metrics,
        "threshold": threshold,
        "X_test_features": X_test_flat,
        "y_test": test_y,
        "y_prob": y_prob_test,
    }


def cross_validate_logistic(X, y, n_splits: int = 5):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []

    for train_idx, test_idx in kf.split(X, y):
        result = fit_logistic(X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        fold_metrics.append(result["metrics"])

    return _aggregate_metrics(fold_metrics)


def _aggregate_metrics(fold_metrics: list[dict]) -> dict:
    keys = ["f1_default", "recall_default", "precision_default", "roc_auc", "pr_auc", "accuracy"]
    summary = {}
    for key in keys:
        values = [m[key] for m in fold_metrics if key in m and not np.isnan(m[key])]
        summary[f"{key}_mean"] = float(np.mean(values)) if values else np.nan
        summary[f"{key}_std"] = float(np.std(values)) if values else np.nan
    return summary

from __future__ import annotations

import warnings
from itertools import product

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from .config import (
    HYBRID_LSTM_RANDOM_TRIALS,
    HYBRID_OUTER_FOLDS,
    HYBRID_XGB_PARAM_GRID,
    RANDOM_STATE,
)
from .metrics import tune_threshold

warnings.filterwarnings("ignore", category=FutureWarning)

LSTM_SEARCH_SPACE = {
    "lstm_units": [32, 64, 96],
    "dropout": [0.2, 0.3, 0.4],
    "dense_units": [16, 32],
    "learning_rate": [0.001, 0.0005],
    "epochs": [30, 40],
    "batch_size": [32],
}


def smote_sequences(X_seq: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if y.sum() < 2:
        return X_seq, y

    seq_len = X_seq.shape[1]
    n_features = X_seq.shape[2]
    X_flat = X_seq.reshape(len(X_seq), -1)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_flat, y)
    return X_res.reshape(len(X_res), seq_len, n_features), y_res


def sample_lstm_configs(n_trials: int, rng: np.random.Generator) -> list[dict]:
    keys = list(LSTM_SEARCH_SPACE.keys())
    configs = []
    seen = set()

    while len(configs) < n_trials:
        cfg = {}
        for key in keys:
            value = rng.choice(LSTM_SEARCH_SPACE[key])
            if key in {"lstm_units", "dense_units", "epochs", "batch_size"}:
                value = int(value)
            elif key == "learning_rate":
                value = float(value)
            elif key == "dropout":
                value = float(value)
            cfg[key] = value
        signature = tuple(cfg[key] for key in keys)
        if signature in seen:
            continue
        seen.add(signature)
        configs.append(cfg)

    return configs


def _class_weights(y: np.ndarray) -> dict[int, float]:
    positives = max(int(y.sum()), 1)
    negatives = max(len(y) - positives, 1)
    return {0: 1.0, 1: negatives / positives}


def build_lstm_attention_encoder(input_shape: tuple[int, int], params: dict):
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Lambda, Softmax
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    import tensorflow.keras.backend as K

    inputs = Input(shape=input_shape)
    lstm_out = LSTM(params["lstm_units"], return_sequences=True, name="lstm_layer")(inputs)

    attention_scores = Dense(1, activation="tanh", name="attention_scores")(lstm_out)
    attention_scores = Lambda(lambda x: K.squeeze(x, axis=-1))(attention_scores)
    attention_weights = Softmax(axis=1, name="attention_weights")(attention_scores)
    attention_weights_exp = Lambda(lambda x: K.expand_dims(x, axis=-1))(attention_weights)
    context = Lambda(
        lambda tensors: K.sum(tensors[0] * tensors[1], axis=1),
        name="attention_context",
    )([lstm_out, attention_weights_exp])

    x = Dropout(params["dropout"])(context)
    x = Dense(params["dense_units"], activation="relu")(x)
    outputs = Dense(1, activation="sigmoid", name="default_probability")(x)
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=params["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
    return model, early_stop


def attention_context_model(lstm_model):
    from tensorflow.keras.models import Model

    return Model(
        inputs=lstm_model.input,
        outputs=lstm_model.get_layer("attention_context").output,
    )


def fit_lstm_encoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
    use_smote: bool = True,
):
    if use_smote:
        X_train, y_train = smote_sequences(X_train, y_train)

    model, early_stop = build_lstm_attention_encoder(
        (X_train.shape[1], X_train.shape[2]),
        params,
    )
    model.fit(
        X_train,
        y_train,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=0,
        shuffle=False,
        class_weight=_class_weights(y_train),
    )
    return model


def extract_attention_context(lstm_model, X_seq: np.ndarray) -> np.ndarray:
    encoder = attention_context_model(lstm_model)
    return encoder.predict(X_seq, verbose=0)


def _xgb_param_dict(grid_params: dict) -> dict:
    return {
        **grid_params,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "early_stopping_rounds": 25,
    }


def smote_features(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if y.sum() < 2:
        return X, y
    smote = SMOTE(random_state=RANDOM_STATE)
    return smote.fit_resample(X, y)


def fit_hybrid_xgb(
    X_train_embed: np.ndarray,
    y_train: np.ndarray,
    X_eval_embed: np.ndarray,
    y_eval: np.ndarray,
    xgb_params: dict,
):
    X_res, y_res = smote_features(X_train_embed, y_train)
    scale_pos_weight = max(1.0, (len(y_res) - y_res.sum()) / max(y_res.sum(), 1))

    model = XGBClassifier(
        **_xgb_param_dict(xgb_params),
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_res, y_res, eval_set=[(X_eval_embed, y_eval)], verbose=False)
    return model


def refit_hybrid_xgb(
    X_train_embed: np.ndarray,
    y_train: np.ndarray,
    xgb_params: dict,
    n_estimators: int,
):
    X_res, y_res = smote_features(X_train_embed, y_train)
    scale_pos_weight = max(1.0, (len(y_res) - y_res.sum()) / max(y_res.sum(), 1))
    fit_params = {
        **{k: v for k, v in _xgb_param_dict(xgb_params).items() if k != "early_stopping_rounds"},
        "n_estimators": max(50, int(n_estimators)),
        "scale_pos_weight": scale_pos_weight,
    }
    model = XGBClassifier(**fit_params)
    model.fit(X_res, y_res, verbose=False)
    return model


def nested_cv_select_hybrid_config(
    X_train_seq: np.ndarray,
    y_train: np.ndarray,
    outer_folds: int = HYBRID_OUTER_FOLDS,
    lstm_trials: int = HYBRID_LSTM_RANDOM_TRIALS,
    xgb_grid: list[dict] | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    rng = rng or np.random.default_rng(RANDOM_STATE)
    xgb_grid = xgb_grid or HYBRID_XGB_PARAM_GRID
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=RANDOM_STATE)
    lstm_configs = sample_lstm_configs(lstm_trials, rng)

    best = {
        "mean_f1": -1.0,
        "lstm_params": lstm_configs[0],
        "xgb_params": xgb_grid[0],
        "thresholds": [0.5],
    }

    for lstm_params in lstm_configs:
        for xgb_params in xgb_grid:
            fold_f1 = []
            fold_thresholds = []

            for train_idx, val_idx in outer_cv.split(X_train_seq, y_train):
                X_tr = X_train_seq[train_idx]
                y_tr = y_train[train_idx]
                X_val = X_train_seq[val_idx]
                y_val = y_train[val_idx]

                inner_tr_idx, inner_val_idx = _inner_split_indices(len(X_tr), y_tr)
                X_inner_tr = X_tr[inner_tr_idx]
                y_inner_tr = y_tr[inner_tr_idx]
                X_inner_val = X_tr[inner_val_idx]
                y_inner_val = y_tr[inner_val_idx]

                lstm_model = fit_lstm_encoder(
                    X_inner_tr,
                    y_inner_tr,
                    X_inner_val,
                    y_inner_val,
                    lstm_params,
                    use_smote=True,
                )

                embed_tr = extract_attention_context(lstm_model, X_tr)
                embed_val = extract_attention_context(lstm_model, X_val)

                xgb_tr_idx, xgb_val_idx = _inner_split_indices(len(X_tr), y_tr)
                xgb_model = fit_hybrid_xgb(
                    embed_tr[xgb_tr_idx],
                    y_tr[xgb_tr_idx],
                    embed_tr[xgb_val_idx],
                    y_tr[xgb_val_idx],
                    xgb_params,
                )

                y_prob_val = xgb_model.predict_proba(embed_val)[:, 1]
                threshold, f1 = tune_threshold(y_val, y_prob_val, metric="f1")
                fold_f1.append(f1)
                fold_thresholds.append(threshold)

            mean_f1 = float(np.mean(fold_f1))
            if mean_f1 > best["mean_f1"]:
                best = {
                    "mean_f1": mean_f1,
                    "lstm_params": lstm_params,
                    "xgb_params": xgb_params,
                    "thresholds": fold_thresholds,
                }

    best["threshold"] = float(np.median(best["thresholds"]))
    return best


def _inner_split_indices(n: int, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(np.unique(y)) < 2 or n < 8:
        split = max(1, int(n * 0.8))
        idx = np.arange(n)
        return idx[:split], idx[split:]

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, val_idx = next(inner_cv.split(np.zeros(n), y))
    return train_idx, val_idx


def gridsearch_hybrid_xgb_configs() -> list[dict]:
    depths = [3, 4, 5]
    learning_rates = [0.03, 0.04, 0.05]
    n_estimators = [300, 400]
    configs = []
    for depth, lr, trees in product(depths, learning_rates, n_estimators):
        configs.append(
            {
                "max_depth": depth,
                "learning_rate": lr,
                "n_estimators": trees,
            }
        )
    return configs

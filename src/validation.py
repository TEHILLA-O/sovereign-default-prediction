from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, VALIDATION_FRACTION


def stratified_train_val_split(X, y, val_fraction: float = VALIDATION_FRACTION):
    if len(np.unique(y)) < 2 or len(y) < 10:
        split = max(1, int(len(y) * (1 - val_fraction)))
        return X[:split], X[split:], y[:split], y[split:]

    return train_test_split(
        X,
        y,
        test_size=val_fraction,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def calibrate_probabilities(y_val, y_prob_val, y_prob_test):
    if len(np.unique(y_val)) < 2:
        return y_prob_test

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_prob_val, y_val)
    return calibrator.predict(y_prob_test)

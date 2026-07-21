from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, RANDOM_STATE, SEQUENCE_LENGTH, TEST_COUNTRY_FRACTION
from .panel_data import load_merged_dataset


def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    if path == DATA_PATH:
        return load_merged_dataset()
    df = pd.read_csv(path)
    return df.sort_values(["Country", "Year"]).reset_index(drop=True)


def split_countries(df: pd.DataFrame, test_size: float = TEST_COUNTRY_FRACTION):
    country_default = df.groupby("Country")["Default"].max()
    countries = country_default.index.to_numpy()
    labels = country_default.to_numpy()

    train_countries, test_countries = train_test_split(
        countries,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    return set(train_countries), set(test_countries)


def tabular_split(df: pd.DataFrame, features: list[str], train_countries, test_countries):
    train_df = df[df["Country"].isin(train_countries)].copy()
    test_df = df[df["Country"].isin(test_countries)].copy()

    X_train = train_df[features].to_numpy(dtype=float)
    y_train = train_df["Default"].to_numpy(dtype=int)
    X_test = test_df[features].to_numpy(dtype=float)
    y_test = test_df["Default"].to_numpy(dtype=int)

    return X_train, X_test, y_train, y_test, train_df, test_df


def build_sequences(
    df: pd.DataFrame,
    features: list[str],
    sequence_length: int = SEQUENCE_LENGTH,
):
    X_list, y_list, meta = [], [], []

    for country, country_df in df.groupby("Country"):
        country_df = country_df.sort_values("Year")
        values = country_df[features].to_numpy(dtype=float)
        targets = country_df["Default"].to_numpy(dtype=int)
        years = country_df["Year"].to_numpy()

        for idx in range(len(country_df) - sequence_length):
            X_list.append(values[idx : idx + sequence_length])
            y_list.append(targets[idx + sequence_length])
            meta.append({"Country": country, "Year": int(years[idx + sequence_length])})

    return np.asarray(X_list), np.asarray(y_list), meta


def temporal_split(df, features: list[str], cutoff_year: int = 2010):
    train_df = df[df["Year"] <= cutoff_year].copy()
    test_df = df[df["Year"] > cutoff_year].copy()

    X_train = train_df[features].to_numpy(dtype=float)
    y_train = train_df["Default"].to_numpy(dtype=int)
    X_test = test_df[features].to_numpy(dtype=float)
    y_test = test_df["Default"].to_numpy(dtype=int)
    return X_train, X_test, y_train, y_test, train_df, test_df


def sequence_split(df: pd.DataFrame, features: list[str], train_countries, test_countries):
    train_df = df[df["Country"].isin(train_countries)]
    test_df = df[df["Country"].isin(test_countries)]

    X_train, y_train, _ = build_sequences(train_df, features)
    X_test, y_test, test_meta = build_sequences(test_df, features)
    return X_train, X_test, y_train, y_test, test_meta

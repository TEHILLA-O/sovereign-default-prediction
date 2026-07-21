from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "extras" / "sovereign_default_dataset_1980_2022.csv - sovereign_default_dataset_1980_2022.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_STATE = 42
TEST_COUNTRY_FRACTION = 0.2
VALIDATION_FRACTION = 0.2
SEQUENCE_LENGTH = 5

ALL_FEATURES = [
    "Debt_GDP",
    "ExtDebt_GDP",
    "DebtServ_XGDP",
    "RealGDP_growth",
    "GDP_per_capita_USD",
    "Inflation",
    "fiscal_balance",
    "primary_balance",
    "CurrentAccount_GDP",
    "Reserves_months",
    "ExchangeRate_change",
    "Trade_openness",
    "US_FedFundsRate",
    "World_GDP_growth",
    "Oil_price_index",
    "VIX",
    "ICRG_political",
    "ElectionYear",
]

SELECTED_FEATURES = [
    "ElectionYear",
    "World_GDP_growth",
    "Oil_price_index",
    "fiscal_balance",
    "ICRG_political",
    "RealGDP_growth",
    "US_FedFundsRate",
    "VIX",
    "ExtDebt_GDP",
    "Trade_openness",
    "Inflation",
    "DebtServ_XGDP",
]

XGB_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 400,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "early_stopping_rounds": 25,
}

HYBRID_XGB_PARAMS = {
    **XGB_PARAMS,
    "max_depth": 5,
    "learning_rate": 0.04,
    "n_estimators": 500,
}

HYBRID_OUTER_FOLDS = 3
HYBRID_LSTM_RANDOM_TRIALS = 5
HYBRID_XGB_PARAM_GRID = [
    {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 300},
    {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 400},
    {"max_depth": 5, "learning_rate": 0.04, "n_estimators": 400},
    {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 500},
    {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 300},
    {"max_depth": 3, "learning_rate": 0.04, "n_estimators": 400},
]

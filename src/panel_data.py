from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .config import ALL_FEATURES, DATA_PATH, PANEL_FINAL_CANDIDATES, PROJECT_ROOT

TARGET = "Default"
LAST_DATA_SOURCE: dict = {}

PANEL_COLUMN_MAP = {
    "country": "Country",
    "country_name": "country_name",
    "year": "Year",
    "debt_gdp": "Debt_GDP",
    "gdp_growth": "RealGDP_growth",
    "inflation_cpi": "Inflation",
    "trade_gdp": "Trade_openness",
    "reserves_months": "Reserves_months",
    "fed_funds_rate": "US_FedFundsRate",
    "brent_usd_bbl": "Oil_price_index",
    "default": TARGET,
    "Debt_GDP": "Debt_GDP",
    "RealGDP_growth": "RealGDP_growth",
    "Inflation": "Inflation",
    "Trade_openness": "Trade_openness",
    "Reserves_months": "Reserves_months",
    "US_FedFundsRate": "US_FedFundsRate",
    "Oil_price_index": "Oil_price_index",
    TARGET: TARGET,
    "Year": "Year",
    "Country": "Country",
}

WGI_COLUMNS = [
    "wgi_voice",
    "wgi_stability",
    "wgi_effectiveness",
    "wgi_regulatory",
    "wgi_ruleoflaw",
    "wgi_corruption",
]

ISO_FALLBACK = {
    "NGA": "Nigeria",
    "GBR": "United Kingdom",
    "USA": "United States",
    "DEU": "Germany",
    "FRA": "France",
    "BRA": "Brazil",
    "MEX": "Mexico",
    "ARG": "Argentina",
    "GRC": "Greece",
    "ZAF": "South Africa",
}


def find_panel_final() -> Path | None:
    for path in PANEL_FINAL_CANDIDATES:
        if path.is_file() and path.stat().st_size > 50:
            return path
    return None


def find_fallback_dataset() -> Path:
    if DATA_PATH.is_file():
        return DATA_PATH
    search_roots = [PROJECT_ROOT, PROJECT_ROOT / "extras", PROJECT_ROOT / "hybrid", PROJECT_ROOT / "xgb"]
    for base in search_roots:
        hits = list(base.glob("sovereign_default_dataset*.csv"))
        if hits:
            return hits[0]
    raise FileNotFoundError("sovereign_default_dataset CSV not found.")


def find_dataset() -> Path:
    panel = find_panel_final()
    return panel if panel is not None else find_fallback_dataset()


def get_data_source() -> dict:
    return dict(LAST_DATA_SOURCE)


def _iso_to_country_name(value: str) -> str:
    if not isinstance(value, str) or len(value) != 3 or not value.isalpha():
        return value
    code = value.upper()
    try:
        import pycountry

        match = pycountry.countries.get(alpha_3=code)
        if match:
            return match.name
    except ImportError:
        pass
    return ISO_FALLBACK.get(code, value)


def country_match_key(name) -> str:
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return ""
    text = str(name).strip()
    if len(text) == 3 and text.isalpha():
        text = _iso_to_country_name(text)
    return text.casefold()


def normalize_panel_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    rename = {}
    for col in out.columns:
        key = col if col in PANEL_COLUMN_MAP else col.lower()
        if key in PANEL_COLUMN_MAP:
            rename[col] = PANEL_COLUMN_MAP[key]
    out = out.rename(columns=rename)

    if "Year" not in out.columns and "year" in out.columns:
        out = out.rename(columns={"year": "Year"})
    if "Country" not in out.columns and "country" in out.columns:
        out = out.rename(columns={"country": "Country"})

    if "Country" in out.columns:
        out["Country"] = out["Country"].astype(str).map(_iso_to_country_name)

    wgi_present = [c for c in WGI_COLUMNS if c in out.columns]
    if wgi_present and "ICRG_political" not in out.columns:
        out["ICRG_political"] = out[wgi_present].mean(axis=1) * 100

    if "Year" in out.columns:
        out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")

    return out


def _apply_panel_overrides(fallback: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    fb = fallback.copy()
    pn = panel.copy()
    fb["_key"] = fb["Country"].map(country_match_key) + "|" + fb["Year"].astype(str)
    pn["_key"] = pn["Country"].map(country_match_key) + "|" + pn["Year"].astype(str)

    panel_idx = pn.set_index("_key")
    filled = []
    for col in list(ALL_FEATURES) + [TARGET]:
        if col not in panel_idx.columns:
            continue
        updates = panel_idx[col].dropna()
        if updates.empty:
            continue
        key_to_val = updates.to_dict()
        mask = fb["_key"].isin(key_to_val)
        if mask.any():
            fb.loc[mask, col] = fb.loc[mask, "_key"].map(key_to_val)
            filled.append(col)

    return fb.drop(columns=["_key"]), filled


def load_merged_dataset() -> pd.DataFrame:
    fallback_path = find_fallback_dataset()
    fallback = pd.read_csv(fallback_path)
    fallback = fallback.sort_values(["Country", "Year"]).reset_index(drop=True)

    required = {"Country", "Year", TARGET, *ALL_FEATURES}
    missing_fb = required - set(fallback.columns)
    if missing_fb:
        raise ValueError(f"Fallback dataset missing columns: {sorted(missing_fb)}")

    panel_path = find_panel_final()
    if panel_path is None:
        LAST_DATA_SOURCE.clear()
        LAST_DATA_SOURCE.update(
            {
                "primary": "sovereign_default_csv",
                "path": str(fallback_path),
                "panel_final": False,
                "rows": len(fallback),
                "countries": int(fallback["Country"].nunique()),
            }
        )
        return fallback

    panel = normalize_panel_columns(pd.read_csv(panel_path))
    merged, filled_from_panel = _apply_panel_overrides(fallback, panel)

    still_missing = required - set(merged.columns)
    if still_missing:
        raise ValueError(
            f"After merging panel_final with fallback, still missing: {sorted(still_missing)}"
        )

    LAST_DATA_SOURCE.clear()
    LAST_DATA_SOURCE.update(
        {
            "primary": "panel_final",
            "path": str(panel_path),
            "fallback_path": str(fallback_path),
            "panel_final": True,
            "rows": len(merged),
            "countries": int(merged["Country"].nunique()),
            "columns_filled_from_panel": filled_from_panel,
        }
    )

    return merged.sort_values(["Country", "Year"]).reset_index(drop=True)

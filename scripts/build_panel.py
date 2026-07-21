from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ALL_FEATURES, DATA_OUT_DIR, PANEL_FINAL_PATH, PROJECT_ROOT
from src.panel_data import TARGET, find_fallback_dataset, normalize_panel_columns

OUT_DIR = DATA_OUT_DIR
PANEL_PATH = PANEL_FINAL_PATH
START_YEAR, END_YEAR = 1980, 2022

WDI = {
    "GC.DOD.TOTL.GD.ZS": "debt_gdp",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "FP.CPI.TOTL.ZG": "inflation_cpi",
    "NE.TRD.GNFS.ZS": "trade_gdp",
    "FI.RES.TOTL.MO": "reserves_months",
    "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
    "BN.CAB.XOKA.GD.ZS": "current_account_gdp",
    "DT.TDS.DECT.EX.ZS": "debt_serv_xgdp",
    "DT.DOD.DECT.EX.ZS": "ext_debt_gdp",
}


def fetch_wdi() -> pd.DataFrame:
    import wbgapi as wb

    print("Fetching World Bank indicators via wbgapi...")
    economies = [e["id"] for e in wb.economy.list() if not e.get("aggregate")]
    raw = wb.data.DataFrame(
        list(WDI.keys()),
        economy=economies,
        time=range(START_YEAR, END_YEAR + 1),
        numericTimeKeys=True,
        labels=True,
    ).reset_index()

    id_cols = [c for c in ["economy", "time", "Country", "Time"] if c in raw.columns]
    value_cols = [c for c in raw.columns if c in WDI]
    long = raw.melt(id_vars=id_cols, value_vars=value_cols, var_name="indicator", value_name="value")
    long["series"] = long["indicator"].map(WDI)
    year_col = "Time" if "Time" in long.columns else "time"
    panel = (
        long.groupby(["Country", year_col, "series"], as_index=False)["value"]
        .mean()
        .pivot_table(index=["Country", year_col], columns="series", values="value")
        .reset_index()
        .rename(columns={year_col: "year"})
    )
    return panel


def load_sovereign_backbone() -> pd.DataFrame:
    df = pd.read_csv(find_fallback_dataset())
    return df.sort_values(["Country", "Year"]).reset_index(drop=True)


def main() -> Path:
    OUT_DIR.mkdir(exist_ok=True)
    sovereign = load_sovereign_backbone()

    try:
        wb_panel = fetch_wdi()
        print(f"  WB rows: {len(wb_panel)}, countries: {wb_panel['Country'].nunique()}")
        panel = wb_panel.rename(columns={"year": "Year"})
    except Exception as exc:
        print(f"  WB fetch failed ({exc}); exporting enriched sovereign panel.")
        panel = sovereign.copy()

    labels = sovereign[["Country", "Year", TARGET]].drop_duplicates()
    panel = panel.drop(columns=[TARGET], errors="ignore").merge(
        labels, on=["Country", "Year"], how="left"
    )
    panel[TARGET] = panel[TARGET].fillna(0).astype(int)

    sov_feats = sovereign[["Country", "Year"] + [c for c in ALL_FEATURES if c in sovereign.columns]]
    panel = panel.merge(sov_feats, on=["Country", "Year"], how="left", suffixes=("", "_sov"))
    for col in ALL_FEATURES:
        sc = f"{col}_sov"
        if sc in panel.columns:
            panel[col] = panel[col].combine_first(panel[sc]) if col in panel.columns else panel[sc]
            panel = panel.drop(columns=[sc])

    panel = normalize_panel_columns(panel)
    export = ["Country", "Year"] + [c for c in ALL_FEATURES if c in panel.columns] + [TARGET]
    panel[[c for c in export if c in panel.columns]].to_csv(PANEL_PATH, index=False)
    print(f"Saved {PANEL_PATH} ({len(panel)} rows, {panel['Country'].nunique()} countries)")
    return PANEL_PATH


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    main()

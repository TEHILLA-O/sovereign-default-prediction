import pandas as pd

from src.panel_data import find_fallback_dataset, find_panel_final, load_merged_dataset


def test_find_fallback_dataset_exists():
    path = find_fallback_dataset()
    assert path.is_file()
    assert "sovereign_default" in path.name.lower()


def test_load_merged_dataset_has_required_columns():
    df = load_merged_dataset()
    for col in ["Country", "Year", "Default", "Debt_GDP", "RealGDP_growth"]:
        assert col in df.columns
    assert len(df) > 100
    assert df["Country"].nunique() >= 10


def test_find_panel_final_present():
    path = find_panel_final()
    assert path is not None
    assert path.name == "panel_final.csv"


def test_load_data_uses_panel_when_present(tmp_path, monkeypatch):
    import src.panel_data as panel_data

    panel_path = tmp_path / "panel_final.csv"
    pd.read_csv(find_fallback_dataset()).head(80).to_csv(panel_path, index=False)
    monkeypatch.setattr(panel_data, "PANEL_FINAL_CANDIDATES", [panel_path])
    panel_data.LAST_DATA_SOURCE.clear()
    load_merged_dataset()
    assert panel_data.LAST_DATA_SOURCE.get("panel_final") is True

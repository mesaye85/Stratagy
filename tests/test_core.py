import json
import pandas as pd
import pytest
from jsonschema import validate
from main import load_structural_democracy_data, compute_democracy_status, load_weights

SCHEMA_PATH = "schema/schema.json"

def test_schema_compliance():
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    sample = {
        "Country": "Testland",
        "Leadership_Turnover_Score": 8,
        "Local_Governance_Score": 7,
        "Policy_Responsiveness_Score": 6,
        "Democracy_Status": 0
    }
    validate(instance=sample, schema=schema)

def test_aggregation_logic():
    row = {
        "Leadership_Turnover_Score": 8,
        "Local_Governance_Score": 6,
        "Policy_Responsiveness_Score": 4
    }
    weights = {
        "Leadership_Turnover_Score": 0.5,
        "Local_Governance_Score": 0.3,
        "Policy_Responsiveness_Score": 0.2
    }
    expected = 8*0.5 + 6*0.3 + 4*0.2
    assert abs(compute_democracy_status(row, weights) - expected) < 1e-6

def test_input_output(tmp_path):
    csv_path = tmp_path / "sample.csv"
    json_path = tmp_path / "sample.json"
    data = [{
        "Country": "Testland",
        "Leadership_Turnover_Score": 8,
        "Local_Governance_Score": 7,
        "Policy_Responsiveness_Score": 6,
        "Democracy_Status": 0
    }]
    pd.DataFrame(data).to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(data, f)
    df_csv = load_structural_democracy_data(str(csv_path))
    df_json = load_structural_democracy_data(str(json_path))
    assert df_csv.shape == df_json.shape
    assert "Country" in df_csv.columns
    assert "Country" in df_json.columns 
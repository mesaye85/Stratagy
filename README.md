# CivicHorizon: Structural Input Normalization for Political System Assessment

## Overview

**CivicHorizon** is a lightweight, schema-driven tool for evaluating the internal structure and functionality of political systems using a defined set of normalized, manually-verifiable inputs.

The project is designed as an **open framework** to:
- Test assumptions about governance structures
- Normalize qualitative or semi-structured data into comparable values
- Explore how structural features affect internal system resilience, adaptability, and coherence

This tool is intended primarily for:
- Systems designers and political engineers
- Analysts modeling sociopolitical stability
- Developers experimenting with logic-based civic diagnostics
- Educators teaching complex system abstraction

It does **not** use opaque ML models or externally imposed scores. All output is traceable to input.

---

## Recent Technical Improvements

- **Modular Input Pipeline:**
  - Supports both CSV and JSON input formats out of the box.
  - Easily extensible to new formats (e.g., API, NLP, database) via a simple registration decorator.

- **Weighted and User-Configurable Aggregation:**
  - Aggregates indicator fields into the final `Democracy_Status` using user-specified weights (via `weights.json`) or defaults to equal weighting.
  - Weights are normalized and can be changed without code modification.

- **Schema-Driven Validation:**
  - All data is validated against `schema/schema.json`.
  - The loader warns about missing or extra fields, ensuring data quality and schema consistency.

- **Metadata and Auditability:**
  - Each indicator can include `*_definition` and `*_source` fields for operational definitions and data sources.
  - These metadata fields are preserved in all outputs (CSV/JSON) and are not used in scoring.

- **Diagnostic and Field Impact Reporting:**
  - CLI and JSON outputs include a breakdown of each field's contribution to the final score.
  - Reports highlight the highest/lowest impact fields, missing/default values, and provide transparent diagnostics for each country.

- **Experimental Inference Model (Imputation):**
  - Enable with `CIVIC_IMPUTE=1` (environment variable).
  - Trains a lightweight, traceable linear model to fill missing indicator values using available data.
  - Imputed values are clearly marked in the output for auditability.
  - Skips imputation for fields with no data and never fails on incomplete input.

- **Robust Error Handling:**
  - Handles `None`/missing values gracefully in all calculations.
  - Skips model training for fields with all values missing.
  - Aggregation logic never fails due to incomplete or partial data.

- **Improved Test Coverage:**
  - (Described for maintainers) Tests cover schema compliance, aggregation logic, input/output handling, and edge cases (missing fields, empty input, etc.).

---

## Project Objectives

- Define a flexible, low-footprint schema for collecting civic structure indicators
- Enable scoring of countries based on real-world events and system signals, not political alignment
- Support modular input pipelines (manual, scripted, or future NLP/ML)
- Provide auditability of all computed values
- Allow end-users to modify weights, rules, or schema for alternative simulations

---

## How It Works

1. **Input Normalization**  
   Users enter structured data in JSON or CSV format, corresponding to observable civic indicators.

2. **Scoring Engine**  
   Inputs are mapped to a 0–10 scale based on defined logic. Each field corresponds to a structural attribute.

3. **Aggregation**  
   Weighted or equal-weighted aggregation is performed. Final outputs are available in CLI or structured JSON form.

4. **Optionally**  
   A lightweight inference model can be trained to project missing values from partial data. This is experimental and traceable.

---

## Input Schema (Overview)

Each country's evaluation is based on explicitly scored indicators across multiple domains.

| Field | Type | Description |
|-------|------|-------------|
| `Country` | `str` | Country name |
| `Leadership_Turnover_Score` | `number` (0–10) | Score measuring leadership turnover and transition |
| `Local_Governance_Score` | `number` (0–10) | Score measuring effectiveness of local governance |
| `Policy_Responsiveness_Score` | `number` (0–10) | Score measuring how responsive policies are to public needs |
| `Language_Alignment_Score` | `number` (0–10) | Score measuring alignment between official and public language |
| `Participation_Accessibility_Score` | `number` (0–10) | Score measuring accessibility of civic participation |
| `Civilian_Control_Score` | `number` (0–10) | Score measuring civilian control over military and security forces |
| `Structural_Advocacy_and_Prosperity_Score` | `number` (0–10) | Score measuring structural support for advocacy and prosperity |
| `Healthcare_Access_Score` | `number` (0–10) | Score measuring access to healthcare |
| `Education_Completion_Score` | `number` (0–10) | Score measuring education completion rates |
| `Food_Sovereignty_Score` | `number` (0–10) | Score measuring food sovereignty and security |
| `Migration_Freedom_Score` | `number` (0–10) | Score measuring freedom of migration |
| `Agency_Projection_Score` | `number` (0–10) | Score measuring individual agency projection |
| `Pathway_to_Influence_Score` | `number` (0–10) | Score measuring pathways to influence in society |
| `Collective_Future_Confidence_Score` | `number` (0–10) | Score measuring collective confidence in the future |
| `System_Adaptability_Score` | `number` (0–10) | Score measuring adaptability of political and social systems |
| `Collective_Momentum_Score` | `number` (0–10) | Score measuring collective momentum for change |
| `Life_Investment_Score` | `number` (0–10) | Score measuring investment in life and future |
| `Exit_vs_Voice_Behavior_Score` | `number` (0–10) | Score measuring tendency to voice concerns vs exit the system |
| `Narrative_Action_Alignment_Score` | `number` (0–10) | Score measuring alignment between narrative and action |
| `Democracy_Status` | `number` (0–10) | Overall democracy status score |

> Full schema, including metadata fields, can be found in `schema/schema.json`.

---

## Output

The program outputs:
- Raw normalized field values
- Weighted total score (if enabled)
- Breakdown of field impact (per-field contributions, top/bottom impact, missing/default warnings)
- Structured report (`JSON`, optionally `CSV`)
- CLI printout with diagnostic interpretation
- (If enabled) Imputed values are flagged in the output

---

## Technical Stack

- Python 3.11+
- Required dependencies: `pandas`, `numpy`, `scikit-learn`, `pycountry`
- Fully compatible with headless environments or lightweight deployment

---

## Sample Workflow

```bash
# Run the main program to process data and generate complete dataset
python main.py

# Enable experimental inference model for missing value imputation
CIVIC_IMPUTE=1 python main.py

# Train a simple model (experimental feature, standalone)
python model.py --train data/ --fields Agency_Projection_Score Collective_Future_Confidence_Score

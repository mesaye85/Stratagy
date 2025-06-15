import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import pycountry
import os
import importlib

"""
Civic Horizon Project - main.py

This application evaluates the sustainability of democracy within nation-states using structured, behavioral,
and future-oriented metrics. It classifies countries as 'Democratic' or 'Non-Democratic' based on their
ability to empower citizens to shape the future.

The framework includes 5 domains:
1. Power Responsiveness
2. Sociopolitical Coherence
3. Material Well-being & Agency
4. Future Perception
5. Behavioral Future Engagement

Each feature is scored from 0 to 10 based on factual, ground-truth evidence. The model avoids
ideological bias and focuses on structural legitimacy and public behavior.
"""

# Modular input pipeline registry
data_loaders = {}

def register_loader(ext):
    """Decorator to register a data loader for a given file extension (e.g., '.csv')."""
    def decorator(func):
        data_loaders[ext] = func
        return func
    return decorator

@register_loader('.csv')
def load_csv(filepath, columns):
    df = pd.read_csv(filepath)
    for col in columns:
        if col not in df.columns:
            df[col] = None
    df = df[columns]
    return df

@register_loader('.json')
def load_json(filepath, columns):
    with open(filepath, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values())
    df = pd.DataFrame(data)
    for col in columns:
        if col not in df.columns:
            df[col] = None
    df = df[columns]
    return df

# Placeholder for future input modules (e.g., API, NLP, etc.)
# @register_loader('.api')
# def load_api(filepath, columns):
#     # Implement API data loading here
#     pass

# Main data loading function using the modular registry
def load_structural_democracy_data(filepath, schema_path="schema/schema.json"):
    """
    Modular data loader. Selects the appropriate loader based on file extension.
    To add a new loader, use the @register_loader decorator above.
    """
    with open(schema_path, 'r') as schema_file:
        schema = json.load(schema_file)
    columns = list(schema["properties"].keys())
    ext = os.path.splitext(filepath)[-1].lower()
    if ext in data_loaders:
        df = data_loaders[ext](filepath, columns)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    # Warn about missing or extra fields
    missing = [col for col in columns if col not in df.columns]
    extra = [col for col in df.columns if col not in columns]
    if missing:
        print(f"WARNING: The following schema fields are missing from input data: {missing}")
    if extra:
        print(f"WARNING: The following fields are in input data but not in schema: {extra}")
    return df

# Get a comprehensive list of all countries in the world
def get_all_countries():
    return [country.name for country in pycountry.countries]

# Load weights from a JSON file if present, otherwise use equal weights
def load_weights(schema_path="schema/schema.json", weights_path="weights.json"):
    # Get indicator fields from schema
    with open(schema_path, 'r') as schema_file:
        schema = json.load(schema_file)
    indicator_fields = [
        k for k in schema["properties"].keys()
        if k not in ["Country", "Democracy_Status"]
    ]
    # Try to load weights.json
    if os.path.exists(weights_path):
        with open(weights_path, 'r') as f:
            weights = json.load(f)
        # Only keep weights for indicator fields
        weights = {k: float(weights.get(k, 1.0)) for k in indicator_fields}
    else:
        # Equal weights
        weights = {k: 1.0 for k in indicator_fields}
    # Normalize weights to sum to 1
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights

# Compute Democracy_Status as weighted sum of indicator fields
def compute_democracy_status(row, weights):
    score = 0.0
    for field, weight in weights.items():
        value = row.get(field, 0)
        if value is None:
            value = 0
        score += value * weight
    return score

# Update prepare_data to compute Democracy_Status from indicators
def prepare_data(df, weights):
    all_countries = get_all_countries()
    existing_countries = set(df['Country'].tolist())
    missing_countries = [country for country in all_countries if country not in existing_countries]
    default_score = 0
    if missing_countries:
        feature_columns = [col for col in df.columns if col not in ['Country', 'Democracy_Status']]
        missing_data = []
        for country in missing_countries:
            country_data = {'Country': country}
            for col in feature_columns:
                country_data[col] = default_score
            # Democracy_Status will be computed below
            missing_data.append(country_data)
        missing_df = pd.DataFrame(missing_data)
        df = pd.concat([df, missing_df], ignore_index=True)
    # Compute Democracy_Status for all rows
    indicator_fields = list(weights.keys())
    df['Democracy_Status'] = df.apply(lambda row: compute_democracy_status(row, weights), axis=1)
    df_complete = df.copy()
    df_complete['Democracy_Category'] = df_complete['Democracy_Status'].apply(
        lambda x: 'Democratic' if x >= 5.0 else 'Non-Democratic'
    )
    df = df.dropna()
    X = df[indicator_fields]
    df['Democracy_Category'] = df['Democracy_Status'].apply(lambda x: 'Democratic' if x >= 5.0 else 'Non-Democratic')
    y = df['Democracy_Category']
    return X, y, df_complete

# Split and scale dataset

def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# Train model

def train_model(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
    clf.fit(X_train, y_train)
    return clf

# Predict outcomes

def predict(clf, X_test):
    return clf.predict(X_test)

# Evaluation: Confusion Matrix

def print_confusion_matrix(cm, labels):
    print("\nConfusion Matrix:")

    # Handle the case where there's only one class
    if len(labels) == 1:
        print(f"Only one class ({labels[0]}) found in the test data.")
        print(f"Correct predictions: {cm[0][0]}")
        return

    # Check the shape of the confusion matrix
    if cm.shape == (1, 1):
        print(f"Only one class found in predictions.")
        print(f"Correct predictions: {cm[0][0]}")
        return

    # Normal case with multiple classes and a 2x2 confusion matrix
    print(f"{'':>12} Predicted {labels[0]}   Predicted {labels[1]}")
    print(f"Actual {labels[0]:>8} {cm[0][0]:>10} {cm[0][1]:>18}")
    print(f"Actual {labels[1]:>8} {cm[1][0]:>10} {cm[1][1]:>18}")

# Evaluation: Accuracy and Report

def print_evaluation(y_test, y_pred):
    print("\nClassification Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Cross Validation

def print_cross_validation_results(accuracies):
    print("\nK-Fold Cross Validation:")
    print(f"Average Accuracy: {accuracies.mean()*100:.2f}%")
    print(f"Standard Deviation: {accuracies.std()*100:.2f}%")

# Grid Search for Hyperparameter Tuning

def print_grid_search_results(best_accuracy, best_params):
    print("\nGrid Search Results:")
    print(f"Best Accuracy: {best_accuracy * 100:.2f}%")
    print(f"Best Parameters: {best_params}")

# Random Search for Exploration

def print_random_search_results(best_accuracy, best_params):
    print("\nRandom Search Results:")
    print(f"Best Accuracy: {best_accuracy * 100:.2f}%")
    print(f"Best Parameters: {best_params}")

# Execute grid search

def run_grid_search(clf, X_train, y_train):
    params = {'n_estimators': [10, 100, 1000], 'criterion': ['entropy', 'gini']}
    search = GridSearchCV(clf, params, scoring='accuracy', cv=2, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_score_, search.best_params_

# Execute random search

def run_random_search(clf, X_train, y_train):
    params = {'n_estimators': [10, 100, 1000], 'criterion': ['entropy', 'gini']}
    # Total parameter space is 3 × 2 = 6, so set n_iter to 6 or less
    search = RandomizedSearchCV(clf, params, scoring='accuracy', cv=2, n_jobs=-1, n_iter=6)
    search.fit(X_train, y_train)
    return search.best_score_, search.best_params_

# Helper to write DataFrame to JSON, including field impact breakdown
def write_json(df, filepath, weights=None):
    """
    Write the DataFrame to JSON. If weights are provided, include a 'field_contributions' breakdown for each country.
    """
    records = df.to_dict(orient="records")
    if weights is not None:
        indicator_fields = list(weights.keys())
        for rec in records:
            contributions = {}
            for field in indicator_fields:
                value = rec.get(field, 0)
                w = weights[field]
                contrib = (value if value is not None else 0) * w
                contributions[field] = {
                    "value": value,
                    "weight": w,
                    "contribution": contrib
                }
            rec["field_contributions"] = contributions
    with open(filepath, 'w') as f:
        json.dump(records, f, indent=2)

# Print breakdown of field contributions for each country, with diagnostics
def print_field_contributions(df, weights, n=5):
    """
    Print a breakdown of field contributions for the first n countries, including which field had the highest/lowest impact and any missing/default values.
    """
    print("\nSample field contribution breakdown (first {} countries):".format(n))
    indicator_fields = list(weights.keys())
    for idx, row in df.head(n).iterrows():
        print(f"\nCountry: {row['Country']}")
        total = row['Democracy_Status']
        contribs = {}
        missing = []
        default = []
        for field in indicator_fields:
            value = row[field]
            w = weights[field]
            contrib = (value if value is not None else 0) * w
            contribs[field] = contrib
            if value is None:
                missing.append(field)
            elif value == 0:
                default.append(field)
            print(f"  {field}: {value} × {w:.3f} = {contrib:.3f}")
        print(f"  Total Democracy_Status: {total:.3f}")
        # Diagnostics
        if contribs:
            max_field = max(contribs, key=contribs.get)
            min_field = min(contribs, key=contribs.get)
            print(f"  Highest impact: {max_field} ({contribs[max_field]:.3f})")
            print(f"  Lowest impact: {min_field} ({contribs[min_field]:.3f})")
        if missing:
            print(f"  WARNING: Missing values for: {missing}")
        if default:
            print(f"  Note: Default (0) values for: {default}")

# Main application logic

def main():
    """
    Main pipeline. Set CIVIC_IMPUTE=1 to enable experimental inference-based imputation of missing values.
    """
    print("\n--- Civic Horizon: Democracy Sustainability Classifier ---")
    weights = load_weights()
    input_file = os.environ.get("CIVIC_INPUT", "seed_structural_democracy_data.csv")
    df = load_structural_democracy_data(input_file)

    # Optional: Experimental inference model for missing value imputation
    if os.environ.get("CIVIC_IMPUTE", "0") == "1":
        print("\n[EXPERIMENTAL] Imputing missing values using inference model...")
        model_mod = importlib.import_module("model")
        # Use all indicator fields as targets
        indicator_fields = list(weights.keys())
        # Train on available data (drop rows with all indicators missing)
        train_data = df.dropna(subset=indicator_fields, how='all')
        models = model_mod.train_inference_model(train_data.to_dict(orient="records"), indicator_fields)
        if models:
            df = model_mod.impute_missing_values(df, models)
            print("Imputation complete. Imputed values are marked with *_imputed columns.")
        else:
            print("Imputation skipped: could not train models.")

    X, y, df_complete = prepare_data(df, weights)
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    clf = train_model(X_train, y_train)
    y_pred = predict(clf, X_test)
    cm = confusion_matrix(y_test, y_pred)
    print_confusion_matrix(cm, clf.classes_)
    print_evaluation(y_test, y_pred)
    accuracies = cross_val_score(clf, X_train, y_train, cv=2)
    print_cross_validation_results(accuracies)
    best_acc_gs, best_params_gs = run_grid_search(clf, X_train, y_train)
    print_grid_search_results(best_acc_gs, best_params_gs)
    best_acc_rs, best_params_rs = run_random_search(clf, X_train, y_train)
    print_random_search_results(best_acc_rs, best_params_rs)
    output_csv = "complete_democracy_data.csv"
    output_json = "complete_democracy_data.json"
    df_complete.to_csv(output_csv, index=False)
    write_json(df_complete, output_json, weights=weights)
    print(f"\nComplete dataset with all countries saved to {output_csv} and {output_json}")
    print(f"Total countries in the output: {len(df_complete)}")
    print(f"Countries with default scores: {len(df_complete) - len(df)}")
    print_field_contributions(df_complete, weights, n=5)

if __name__ == "__main__":
    main()

"""
Civic Horizon - Optional Lightweight Inference Model

This module provides an experimental, traceable inference model for projecting missing values
from partial data in the Civic Horizon framework.

Usage:
    python model.py --train data/ --fields legal_agency_score perceived_trajectory

The model is designed to be:
1. Lightweight - minimal dependencies and computational requirements
2. Traceable - all predictions can be traced back to input data
3. Experimental - clearly marked as an optional, supplementary tool
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any, Optional

# Optional dependencies - only used if available
try:
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def load_data(data_dir: str) -> List[Dict[str, Any]]:
    """Load all JSON files from the specified directory."""
    data = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            with open(os.path.join(data_dir, filename), 'r') as f:
                data.append(json.load(f))
    return data


def train_inference_model(data: List[Dict[str, Any]], target_fields: List[str]):
    """
    Train a simple model to infer missing values for specified fields.
    Only uses numeric indicator fields as features.
    Skips fields where all values are NaN.
    """
    if not SKLEARN_AVAILABLE:
        print("Error: scikit-learn is required for training inference models.")
        print("Install with: pip install scikit-learn pandas numpy")
        return None
    df = pd.DataFrame(data)
    non_features = set(["Country", "Democracy_Status"]) | set([f for f in df.columns if not pd.api.types.is_numeric_dtype(df[f])])
    models = {}
    for target in target_fields:
        if target not in df.columns:
            print(f"Error: Field '{target}' not found in data.")
            return None
        features = [col for col in df.select_dtypes(include=[float, int]).columns if col != target and col not in non_features]
        if not features:
            print(f"Warning: No features available to predict {target}")
            continue
        y = df[target]
        if y.isna().all():
            print(f"Warning: All values missing for {target}; skipping model.")
            continue
        X = df[features].fillna(df[features].mean())
        y = y.fillna(y.mean())
        model = LinearRegression()
        model.fit(X, y)
        models[target] = {
            'model': model,
            'features': features
        }
        y_pred = model.predict(X)
        mse = np.mean((y_pred - y) ** 2)
        print(f"Model for {target}: Mean Squared Error = {mse:.2f}")
    return models


def impute_missing_values(df, models):
    """
    Fill missing values in the DataFrame using the provided trained models.
    Only uses numeric indicator fields as features.
    """
    df = df.copy()
    for target, model_info in models.items():
        model = model_info['model']
        features = model_info['features']
        if target not in df.columns:
            continue
        mask = df[target].isna()
        if not mask.any():
            continue
        X_missing = df.loc[mask, features].fillna(df[features].mean())
        preds = model.predict(X_missing)
        df.loc[mask, target] = preds
        imputed_col = f"{target}_imputed"
        df[imputed_col] = False
        df.loc[mask, imputed_col] = True
    return df


def main():
    """Main function to handle command line arguments and train models."""
    parser = argparse.ArgumentParser(description='Train inference models for Civic Horizon')
    parser.add_argument('--train', type=str, help='Directory containing training data')
    parser.add_argument('--fields', nargs='+', help='Fields to train models for')
    
    args = parser.parse_args()
    
    if not args.train or not args.fields:
        parser.print_help()
        return
    
    print("Civic Horizon - Experimental Inference Model Training")
    print("=" * 50)
    print("Warning: This is an experimental feature. All predictions should be verified.")
    
    # Load data
    data = load_data(args.train)
    if not data:
        print(f"Error: No JSON files found in {args.train}")
        return
    
    print(f"Loaded {len(data)} data files")
    
    # Train models
    models = train_inference_model(data, args.fields)
    if not models:
        return
    
    print("\nTraining complete. Models can now be used to project missing values.")
    print("Remember that all projections are experimental and should be verified.")


if __name__ == "__main__":
    main()
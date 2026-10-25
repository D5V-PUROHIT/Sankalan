"""
================================================================================
SANKALAN AI PLATFORM - DATA PREPROCESSING PIPELINE
File: backend/analytics/data_preprocessing.py
Role: Automated Data Cleaning, Role Classification, Imputation, & VAE Preprocessing
================================================================================

This module implements:
1. File Loading: Automatic format detection (.csv, .xlsx, .xls)
2. Deduplication: Removes duplicate rows from the raw tabular input
3. Column Role Classification:
   - Identifiers / Non-generative keys (high cardinality or constant)
   - Datetime fields (parsed and stored for empirical resampling)
   - Generative Numeric (continuous/discrete scaled with StandardScaler)
   - Generative Categorical (encoded with OneHotEncoder)
4. Robust Imputation:
   - Numeric missing values imputed with median
   - Categorical missing values imputed with "Unknown"
5. Feature Map Creation: Maps input matrix column slice boundaries for composite VAE loss
6. Artifact Persistence: Saves scaler.pkl, ohe_encoders.pkl, column_roles.json, X_train.npy
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os
import json


def clean_and_preprocess(file_path: str, output_dir: str = "backend/data/processed", models_dir: str = "backend/models/saved_models"):
    """
    Cleans raw dataset, classifies column roles, fits standard transformers,
    and produces ready-to-train numpy arrays and feature maps for the VAE model.
    """
    try:
        fp_lower = file_path.lower()
        if fp_lower.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif fp_lower.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please upload CSV or Excel files.")

        df = df.drop_duplicates()
        total_rows = len(df)
        
        column_roles = {
            "identifiers": [],
            "dates": [],
            "generative_numeric": [],
            "generative_categorical": []
        }
        
        # 1. Role Classification
        for col in df.columns:
            # Check for dates first
            is_date = False
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                is_date = True
            elif df[col].dtype == 'object':
                try:
                    # Attempt to parse a sample to see if it's a date
                    sample = df[col].dropna().head(10)
                    if len(sample) > 0:
                        pd.to_datetime(sample)
                        df[col] = pd.to_datetime(df[col])
                        is_date = True
                except (ValueError, TypeError):
                    pass
            
            if is_date:
                column_roles["dates"].append(col)
                continue
                
            # Check for identifiers (high cardinality > 90% unique)
            if total_rows > 0 and (df[col].nunique() / total_rows > 0.9):
                column_roles["identifiers"].append(col)
                continue
                
            # Otherwise, it's a generative feature
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() <= 1:
                    column_roles["identifiers"].append(col)  # Treat constant as identifier
                else:
                    column_roles["generative_numeric"].append(col)
            else:
                if df[col].nunique() <= 1:
                    column_roles["identifiers"].append(col)
                else:
                    column_roles["generative_categorical"].append(col)

        # 2. Imputation
        for col in column_roles["generative_numeric"]:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
                
        for col in column_roles["generative_categorical"]:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna("Unknown")

        # 3. Encoding & Feature Map Building
        os.makedirs(models_dir, exist_ok=True)
        feature_map = []
        processed_features = []
        current_idx = 0
        
        # Process Numeric Columns
        if column_roles["generative_numeric"]:
            scaler = StandardScaler()
            scaled_num = scaler.fit_transform(df[column_roles["generative_numeric"]])
            joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
            
            for i, col in enumerate(column_roles["generative_numeric"]):
                col_data = scaled_num[:, i:i+1]
                processed_features.append(col_data)
                feature_map.append({
                    "name": col,
                    "type": "numeric",
                    "start": current_idx,
                    "end": current_idx + 1
                })
                current_idx += 1
                
        # Process Categorical Columns (One-Hot Encoding)
        encoders = {}
        if column_roles["generative_categorical"]:
            for col in column_roles["generative_categorical"]:
                ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_col = ohe.fit_transform(df[[col]])
                encoders[col] = ohe
                processed_features.append(encoded_col)
                
                num_classes = encoded_col.shape[1]
                feature_map.append({
                    "name": col,
                    "type": "categorical",
                    "start": current_idx,
                    "end": current_idx + num_classes
                })
                current_idx += num_classes
                
            joblib.dump(encoders, os.path.join(models_dir, 'ohe_encoders.pkl'))

        # 4. Save Column Roles and Feature Map
        column_roles["feature_map"] = feature_map
        column_roles["input_dim"] = current_idx
        with open(os.path.join(models_dir, 'column_roles.json'), 'w') as f:
            json.dump(column_roles, f, indent=4)

        # 5. Concatenate & Save Processed Training Matrix
        if processed_features:
            X_train = np.hstack(processed_features)
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        else:
            X_train = np.empty((total_rows, 0))

        # 6. Save Non-Generative Data (IDs and Dates) for Inverse Attachment
        non_gen_cols = column_roles["identifiers"] + column_roles["dates"]
        if non_gen_cols:
            df[non_gen_cols].to_csv(os.path.join(output_dir, 'non_generative_data.csv'), index=False)

        return {
            "status": "success",
            "message": "Data cleaned, classified, and preprocessed successfully.",
            "input_dim": current_idx,
            "feature_map": feature_map,
            "column_roles": column_roles
        }

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return {"status": "error", "message": str(e)}

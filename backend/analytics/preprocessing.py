import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os
import json

def clean_and_preprocess(file_path: str, output_dir: str = "backend/data/processed", models_dir: str = "backend/models/saved_models"):
    """
    Implements Addendum A: Column Role Classification & VAE preprocessing.
    """
    try:
        fp_lower = file_path.lower()
        if fp_lower.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif fp_lower.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format.")

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
                        # If successful, parse the whole column
                        df[col] = pd.to_datetime(df[col])
                        is_date = True
                except (ValueError, TypeError):
                    pass
            
            if is_date:
                column_roles["dates"].append(col)
                continue
                
            # Check for identifiers (high cardinality)
            if df[col].nunique() / total_rows > 0.9:
                column_roles["identifiers"].append(col)
                continue
                
            # Otherwise, it's a generative feature
            if pd.api.types.is_numeric_dtype(df[col]):
                # If a numeric column is constant, we could drop it, but we'll classify it and handle it.
                if df[col].nunique() <= 1:
                    column_roles["identifiers"].append(col) # Treat constant as identifier to ignore it
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
        
        # Process Numeric
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
                
        # Process Categorical
        encoders = {}
        for col in column_roles["generative_categorical"]:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_col = ohe.fit_transform(df[[col]])
            encoders[col] = ohe
            
            num_classes = encoded_col.shape[1]
            processed_features.append(encoded_col)
            feature_map.append({
                "name": col,
                "type": "categorical",
                "start": current_idx,
                "end": current_idx + num_classes
            })
            current_idx += num_classes
            
        if encoders:
            joblib.dump(encoders, os.path.join(models_dir, 'ohe_encoders.pkl'))

        # Save Column Roles & Feature Map
        metadata = {
            "roles": column_roles,
            "feature_map": feature_map,
            "total_generative_dim": current_idx
        }
        with open(os.path.join(models_dir, 'column_roles.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

        # Combine into final training matrix X
        if processed_features:
            X_train = np.hstack(processed_features)
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
            
            # Save a copy of the non-generative data for sampling later
            df[column_roles["dates"] + column_roles["identifiers"]].to_csv(
                os.path.join(output_dir, 'non_generative_data.csv'), index=False
            )

        return {
            "status": "success",
            "roles_assigned": {
                "identifiers": len(column_roles["identifiers"]),
                "dates": len(column_roles["dates"]),
                "numeric": len(column_roles["generative_numeric"]),
                "categorical": len(column_roles["generative_categorical"])
            },
            "vae_input_dim": current_idx
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "status": "failed"}

import numpy as np
import pandas as pd
import joblib
import json
import os
import tensorflow as tf

def generate_and_inverse_transform(vae_model, feature_map, column_roles, num_records, models_dir, raw_df_path):
    """
    Samples from the VAE latent space, decodes, and inverse transforms
    the numeric and one-hot categorical features back to a pandas DataFrame.
    Also handles re-attaching mock identifiers and sampled dates.
    """
    # 1. Sample from latent space N(0, 1)
    latent_dim = vae_model.latent_dim
    z_sample = tf.random.normal(shape=(num_records, latent_dim))
    
    # 2. Decode
    synthetic_scaled = vae_model.decoder(z_sample).numpy()
    
    # 3. Load Transformers
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    encoders_path = os.path.join(models_dir, 'ohe_encoders.pkl')
    
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    encoders = joblib.load(encoders_path) if os.path.exists(encoders_path) else None
    
    synthetic_data = {}
    
    # 4. Inverse Transform based on feature map
    for feature in feature_map:
        name = feature["name"]
        start = feature["start"]
        end = feature["end"]
        ftype = feature["type"]
        
        slice_data = synthetic_scaled[:, start:end]
        
        if ftype == "numeric":
            synthetic_data[name] = slice_data.flatten()
        elif ftype == "categorical":
            # Argmax for one-hot to index
            class_indices = np.argmax(slice_data, axis=-1)
            # Inverse transform index to category string
            ohe = encoders[name]
            # OneHotEncoder categories_ is a list of arrays per feature
            cats = ohe.categories_[0]
            synthetic_data[name] = [cats[i] if i < len(cats) else "Unknown" for i in class_indices]

    # Convert numeric dictionary to dataframe first so we can scale it together
    synth_df = pd.DataFrame(synthetic_data)
    
    numeric_cols = column_roles["generative_numeric"]
    if scaler and numeric_cols:
        synth_df[numeric_cols] = scaler.inverse_transform(synth_df[numeric_cols])
        
        # Preserve domain constraints & discrete column types from original dataset
        try:
            original_df = pd.read_csv(raw_df_path) if raw_df_path.endswith('.csv') else pd.read_excel(raw_df_path)
            for col in numeric_cols:
                if col in original_df.columns:
                    orig_series = pd.to_numeric(original_df[col], errors='coerce').dropna()
                    if len(orig_series) > 0:
                        # If original column is integer or discrete (e.g. SeniorCitizen with 0/1 or tenure)
                        is_int_col = pd.api.types.is_integer_dtype(original_df[col]) or bool((orig_series % 1 == 0).all() and orig_series.nunique() <= 100)
                        if is_int_col:
                            synth_df[col] = synth_df[col].round()
                            synth_df[col] = synth_df[col].clip(lower=orig_series.min(), upper=orig_series.max())
                            synth_df[col] = synth_df[col].astype(int)
                        else:
                            # Clip non-negative floats (e.g., MonthlyCharges, TotalCharges)
                            if orig_series.min() >= 0:
                                synth_df[col] = synth_df[col].clip(lower=0.0)
                            # Generous margin clipping to prevent extreme outlier divergence
                            margin = (orig_series.max() - orig_series.min()) * 0.1
                            synth_df[col] = synth_df[col].clip(lower=orig_series.min() - margin, upper=orig_series.max() + margin)
        except Exception as e:
            print("Domain constraint preservation error:", e)
    
    # 5. Handle Non-Generative Columns (Dates and Identifiers)
    non_gen_path = os.path.join(models_dir, '../data/processed/non_generative_data.csv')
    if os.path.exists(non_gen_path):
        non_gen_df = pd.read_csv(non_gen_path)
        
        # Sample dates from empirical distribution
        for date_col in column_roles.get("dates", []):
            if date_col in non_gen_df.columns:
                sampled_dates = non_gen_df[date_col].dropna().sample(n=num_records, replace=True).values
                synth_df[date_col] = sampled_dates
                
    # Generate sequential mock IDs for identifiers
    for id_col in column_roles.get("identifiers", []):
        synth_df[id_col] = [f"SYN-{i:06d}" for i in range(1, num_records + 1)]
        
    # Reorder columns to match original if possible
    try:
        original_df = pd.read_csv(raw_df_path) if raw_df_path.endswith('.csv') else pd.read_excel(raw_df_path)
        final_cols = [c for c in original_df.columns if c in synth_df.columns]
        synth_df = synth_df[final_cols]
    except:
        pass
        
    return synth_df

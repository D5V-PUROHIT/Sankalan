import pandas as pd
import numpy as np

def is_valid_categorical(df: pd.DataFrame, col: str) -> bool:
    """Filters out IDs, keys, unnamed index columns, and high-cardinality columns from categorical views."""
    col_lower = col.lower()
    if col_lower.startswith('unnamed') or col_lower in ['customerid', 'id', 'uuid', 'guid', 'order_id', 'orderid']:
        return False
    if col_lower.endswith('_id') or col_lower.endswith('id'):
        return False
        
    n_unique = df[col].nunique()
    total_len = len(df)
    
    # Must have at least 2 categories and at most 25 categories
    if n_unique <= 1 or n_unique > 25:
        return False
    if total_len > 0 and (n_unique / total_len > 0.15):
        return False
        
    return True

def generate_eda_data(file_path: str):
    """
    Generates rich exploratory data analysis data for the frontend EDA tab.
    Includes correlation matrix, summary statistics, and clean column choices.
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
            
        raw_num_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        num_cols = [c for c in raw_num_cols if not c.lower().startswith('unnamed') and not c.lower().endswith('_id') and c.lower() not in ['id', 'uuid', 'guid', 'customerid']]
        
        # Filter raw categorical columns to only genuine categories
        raw_cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        cat_cols = [c for c in raw_cat_cols if is_valid_categorical(df, c)]
        
        # If no strict categories found, fallback to anything with nunique <= 30 (excluding unnamed/IDs)
        if not cat_cols:
            cat_cols = [c for c in raw_cat_cols if not c.lower().startswith('unnamed') and 1 < df[c].nunique() <= 30]
        
        # 1. Numerical Correlation Matrix
        correlation_data = None
        if len(num_cols) > 1:
            clean_num_df = pd.DataFrame({c: pd.to_numeric(df[c], errors='coerce') for c in num_cols})
            corr_df = clean_num_df.corr().fillna(0)
            correlation_data = {
                "z": [[round(float(v), 3) for v in row] for row in corr_df.values.tolist()],
                "x": corr_df.columns.tolist(),
                "y": corr_df.columns.tolist()
            }
            
        # 2. Per-column distribution summaries
        column_distributions = {}
        for col in num_cols:
            valid_data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(valid_data) > 0:
                sampled = valid_data.sample(min(3000, len(valid_data)), random_state=42).tolist() if len(valid_data) > 3000 else valid_data.tolist()
                std_val = float(valid_data.std()) if not np.isnan(valid_data.std()) and valid_data.std() > 0 else 0.0
                column_distributions[col] = {
                    "values": [round(float(v), 2) for v in sampled],
                    "mean": round(float(valid_data.mean()), 2) if not np.isnan(valid_data.mean()) else 0.0,
                    "median": round(float(valid_data.median()), 2) if not np.isnan(valid_data.median()) else 0.0,
                    "std": round(std_val, 2),
                    "min": round(float(valid_data.min()), 2) if not np.isnan(valid_data.min()) else 0.0,
                    "max": round(float(valid_data.max()), 2) if not np.isnan(valid_data.max()) else 0.0
                }
                
        # 3. Categorical value counts (top 15 categories max)
        categorical_distributions = {}
        for col in cat_cols:
            val_counts = df[col].astype(str).value_counts().head(15).to_dict()
            categorical_distributions[col] = {
                "labels": list(val_counts.keys()),
                "counts": list(val_counts.values())
            }

        return {
            "status": "success",
            "num_columns": num_cols,
            "cat_columns": cat_cols,
            "correlation": correlation_data,
            "column_distributions": column_distributions,
            "categorical_distributions": categorical_distributions
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


def generate_comparison_data(real_file_path: str, synth_file_path: str, num_col: str = None, cat_col: str = None):
    """
    Generates side-by-side comparison data between Real and Synthetic datasets
    for the Real vs Synthetic Comparison Tab.
    """
    try:
        r_df = pd.read_csv(real_file_path) if real_file_path.endswith('.csv') else pd.read_excel(real_file_path)
        s_df = pd.read_csv(synth_file_path) if synth_file_path.endswith('.csv') else pd.read_excel(synth_file_path)

        # Detect columns common to both datasets, excluding unnamed index and identifier columns
        common_cols = [c for c in r_df.columns if c in s_df.columns]
        num_cols = []
        for c in common_cols:
            c_lower = c.lower()
            if c_lower.startswith('unnamed') or c_lower in ['id', 'uuid', 'guid', 'customerid'] or c_lower.endswith('_id'):
                continue
            r_num = pd.to_numeric(r_df[c], errors='coerce')
            s_num = pd.to_numeric(s_df[c], errors='coerce')
            # Column is considered numeric if both real and synthetic have > 50% non-null numeric values
            if r_num.notnull().mean() > 0.5 and s_num.notnull().mean() > 0.5:
                num_cols.append(c)

        # Detect valid categorical columns
        cat_cols = [c for c in common_cols if c not in num_cols and is_valid_categorical(r_df, c)]
        if not cat_cols:
            cat_cols = [c for c in common_cols if c not in num_cols and not c.lower().startswith('unnamed') and 1 < r_df[c].nunique() <= 30]

        # Selected numeric column comparison
        num_distribution_comparison = None
        target_num = num_col if (num_col and num_col in num_cols) else (num_cols[0] if num_cols else None)
        if target_num:
            r_vals = pd.to_numeric(r_df[target_num], errors='coerce').dropna()
            s_vals = pd.to_numeric(s_df[target_num], errors='coerce').dropna()
            
            if len(r_vals) > 0 and len(s_vals) > 0:
                r_sample = r_vals.sample(min(3000, len(r_vals)), random_state=42).tolist()
                s_sample = s_vals.sample(min(3000, len(s_vals)), random_state=42).tolist()
                
                r_std = float(r_vals.std()) if not np.isnan(r_vals.std()) and r_vals.std() > 0 else 0.0
                s_std = float(s_vals.std()) if not np.isnan(s_vals.std()) and s_vals.std() > 0 else 0.0
                
                num_distribution_comparison = {
                    "column": target_num,
                    "real_values": [round(float(v), 2) for v in r_sample],
                    "synth_values": [round(float(v), 2) for v in s_sample],
                    "real_stats": {
                        "mean": round(float(r_vals.mean()), 2) if not np.isnan(r_vals.mean()) else 0.0,
                        "median": round(float(r_vals.median()), 2) if not np.isnan(r_vals.median()) else 0.0,
                        "std": round(r_std, 2),
                        "min": round(float(r_vals.min()), 2) if not np.isnan(r_vals.min()) else 0.0,
                        "max": round(float(r_vals.max()), 2) if not np.isnan(r_vals.max()) else 0.0
                    },
                    "synth_stats": {
                        "mean": round(float(s_vals.mean()), 2) if not np.isnan(s_vals.mean()) else 0.0,
                        "median": round(float(s_vals.median()), 2) if not np.isnan(s_vals.median()) else 0.0,
                        "std": round(s_std, 2),
                        "min": round(float(s_vals.min()), 2) if not np.isnan(s_vals.min()) else 0.0,
                        "max": round(float(s_vals.max()), 2) if not np.isnan(s_vals.max()) else 0.0
                    }
                }

        # Selected categorical column comparison
        cat_distribution_comparison = None
        target_cat = cat_col if (cat_col and cat_col in cat_cols) else (cat_cols[0] if cat_cols else None)
        if target_cat:
            r_series = r_df[target_cat].dropna().astype(str)
            s_series = s_df[target_cat].dropna().astype(str)
            
            r_pct = r_series.value_counts(normalize=True).to_dict()
            s_pct = s_series.value_counts(normalize=True).to_dict()
            
            # Limit to top 15 labels to prevent illegible charts
            top_labels = r_series.value_counts().head(15).index.tolist()
            if not top_labels:
                top_labels = s_series.value_counts().head(15).index.tolist()
            
            cat_distribution_comparison = {
                "column": target_cat,
                "labels": top_labels,
                "real_pct": [round(float(r_pct.get(l, 0.0) * 100), 1) for l in top_labels],
                "synth_pct": [round(float(s_pct.get(l, 0.0) * 100), 1) for l in top_labels]
            }

        # Correlation matrices (safely computed using purely numeric coerced DataFrames)
        correlation_comparison = None
        if len(num_cols) > 1:
            r_clean = pd.DataFrame({c: pd.to_numeric(r_df[c], errors='coerce') for c in num_cols})
            s_clean = pd.DataFrame({c: pd.to_numeric(s_df[c], errors='coerce') for c in num_cols})
            r_corr = r_clean.corr().fillna(0)
            s_corr = s_clean.corr().fillna(0)
            diff_corr = np.abs(r_corr.values - s_corr.values)
            correlation_comparison = {
                "columns": num_cols,
                "real": [[round(float(v), 3) for v in row] for row in r_corr.values.tolist()],
                "synth": [[round(float(v), 3) for v in row] for row in s_corr.values.tolist()],
                "diff": [[round(float(v), 3) for v in row] for row in diff_corr.tolist()],
                "mae": round(float(np.mean(diff_corr)), 4)
            }

        return {
            "status": "success",
            "available_num_columns": num_cols,
            "available_cat_columns": cat_cols,
            "num_distribution": num_distribution_comparison,
            "cat_distribution": cat_distribution_comparison,
            "correlation_comparison": correlation_comparison
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


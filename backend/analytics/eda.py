import pandas as pd
import numpy as np

def generate_eda_data(file_path: str):
    """
    Generates rich exploratory data analysis data for the frontend EDA tab.
    Includes correlation matrix, summary statistics, and column choices.
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
            
        num_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # 1. Numerical Correlation Matrix
        correlation_data = None
        if len(num_cols) > 1:
            corr_df = df[num_cols].corr().fillna(0)
            correlation_data = {
                "z": [[round(v, 3) for v in row] for row in corr_df.values.tolist()],
                "x": corr_df.columns.tolist(),
                "y": corr_df.columns.tolist()
            }
            
        # 2. Per-column distribution summaries
        column_distributions = {}
        for col in num_cols:
            valid_data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(valid_data) > 0:
                # Sample up to 3000 points to keep payload snappy
                sampled = valid_data.sample(min(3000, len(valid_data)), random_state=42).tolist() if len(valid_data) > 3000 else valid_data.tolist()
                column_distributions[col] = {
                    "values": [round(v, 2) for v in sampled],
                    "mean": round(float(valid_data.mean()), 2),
                    "median": round(float(valid_data.median()), 2),
                    "std": round(float(valid_data.std()), 2) if valid_data.std() > 0 else 0.0,
                    "min": round(float(valid_data.min()), 2),
                    "max": round(float(valid_data.max()), 2)
                }
                
        # 3. Categorical value counts (top 10 categories)
        categorical_distributions = {}
        for col in cat_cols:
            val_counts = df[col].astype(str).value_counts().head(10).to_dict()
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

        num_cols = [c for c in r_df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns if c in s_df.columns]
        cat_cols = [c for c in r_df.select_dtypes(include=['object', 'category', 'bool']).columns if c in s_df.columns]

        # Selected numeric column comparison
        num_distribution_comparison = None
        target_num = num_col if (num_col and num_col in num_cols) else (num_cols[0] if num_cols else None)
        if target_num:
            r_vals = pd.to_numeric(r_df[target_num], errors='coerce').dropna()
            s_vals = pd.to_numeric(s_df[target_num], errors='coerce').dropna()
            
            # Sample for performance if needed
            r_sample = r_vals.sample(min(3000, len(r_vals)), random_state=42).tolist()
            s_sample = s_vals.sample(min(3000, len(s_vals)), random_state=42).tolist()
            
            num_distribution_comparison = {
                "column": target_num,
                "real_values": [round(v, 2) for v in r_sample],
                "synth_values": [round(v, 2) for v in s_sample],
                "real_stats": {
                    "mean": round(float(r_vals.mean()), 2),
                    "median": round(float(r_vals.median()), 2),
                    "std": round(float(r_vals.std()), 2) if r_vals.std() > 0 else 0.0,
                    "min": round(float(r_vals.min()), 2),
                    "max": round(float(r_vals.max()), 2)
                },
                "synth_stats": {
                    "mean": round(float(s_vals.mean()), 2),
                    "median": round(float(s_vals.median()), 2),
                    "std": round(float(s_vals.std()), 2) if s_vals.std() > 0 else 0.0,
                    "min": round(float(s_vals.min()), 2),
                    "max": round(float(s_vals.max()), 2)
                }
            }

        # Selected categorical column comparison
        cat_distribution_comparison = None
        target_cat = cat_col if (cat_col and cat_col in cat_cols) else (cat_cols[0] if cat_cols else None)
        if target_cat:
            r_pct = r_df[target_cat].astype(str).value_counts(normalize=True).to_dict()
            s_pct = s_df[target_cat].astype(str).value_counts(normalize=True).to_dict()
            
            all_labels = sorted(list(set(r_pct.keys()).union(set(s_pct.keys()))))
            cat_distribution_comparison = {
                "column": target_cat,
                "labels": all_labels,
                "real_pct": [round(r_pct.get(l, 0.0) * 100, 1) for l in all_labels],
                "synth_pct": [round(s_pct.get(l, 0.0) * 100, 1) for l in all_labels]
            }

        # Correlation matrices
        correlation_comparison = None
        if len(num_cols) > 1:
            r_corr = r_df[num_cols].corr().fillna(0)
            s_corr = s_df[num_cols].corr().fillna(0)
            diff_corr = np.abs(r_corr.values - s_corr.values)
            correlation_comparison = {
                "columns": num_cols,
                "real": [[round(v, 3) for v in row] for row in r_corr.values.tolist()],
                "synth": [[round(v, 3) for v in row] for row in s_corr.values.tolist()],
                "diff": [[round(v, 3) for v in row] for row in diff_corr.tolist()],
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

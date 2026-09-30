import pandas as pd
import numpy as np

def generate_eda_data(file_path: str):
    """
    Generates data required for the EDA charts on the frontend.
    Returns correlation matrix and distribution data for the first numerical column.
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
            
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # 1. Correlation Matrix
        if len(num_cols) > 1:
            corr_df = df[num_cols].corr().fillna(0)
            z_values = corr_df.values.tolist()
            x_labels = corr_df.columns.tolist()
            y_labels = corr_df.columns.tolist()
            correlation_data = {
                "z": z_values,
                "x": x_labels,
                "y": y_labels
            }
        else:
            correlation_data = None
            
        # 2. Distribution Data (Histogram for the first numeric column)
        distribution_data = None
        if num_cols:
            primary_col = num_cols[0] # Just pick the first numeric col for demonstration
            # Drop NaNs for histogram
            valid_data = df[primary_col].dropna().tolist()
            # If dataset is huge, sample it to avoid massive JSON payloads
            if len(valid_data) > 5000:
                valid_data = df[primary_col].dropna().sample(5000).tolist()
                
            distribution_data = {
                "column_name": primary_col,
                "values": valid_data
            }
            
        return {
            "status": "success",
            "correlation": correlation_data,
            "distribution": distribution_data
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

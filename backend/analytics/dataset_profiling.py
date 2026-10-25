"""
================================================================================
SANKALAN AI PLATFORM - DATASET PROFILING
File: backend/analytics/dataset_profiling.py
Role: Automated Dataset Health, Missingness, Cardinality & Quality Profiler
================================================================================

This module implements:
1. File Format Detection: Handles CSV and Excel (.xlsx, .xls)
2. Structural Dimensions: Number of rows, columns, and memory shape
3. Duplicate Detection: Counts exact row-level duplicates in raw upload
4. Column-Level Profiling:
   - Inferred column type (Numerical vs Categorical)
   - Specific Pandas dtype
   - Null / Missing count and percentage
   - Unique value cardinality
5. Health Profile Report Generation: Emits JSON summary for initial dashboard render
================================================================================
"""

import pandas as pd
import numpy as np


def profile_dataset(file_path: str) -> dict:
    """
    Reads a CSV or Excel dataset and performs automated data profiling.
    Extracts number of rows, columns, data types, missing values, duplicates, and unique counts.
    """
    try:
        fp_lower = file_path.lower()
        if fp_lower.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif fp_lower.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please upload CSV or Excel files.")

        # 1. Basic Shape
        num_rows = len(df)
        num_cols = len(df.columns)

        # 2. Duplicates
        duplicate_count = int(df.duplicated().sum())

        # 3. Column-level statistics
        columns_info = []
        missing_total = 0

        for col in df.columns:
            col_data = df[col]
            missing_count = int(col_data.isnull().sum())
            missing_total += missing_count
            
            # Identify data type (categorical vs numerical)
            dtype = str(col_data.dtype)
            col_type = "Categorical" if dtype == 'object' or pd.api.types.is_categorical_dtype(col_data) else "Numerical"
            
            # Unique values
            unique_count = int(col_data.nunique())

            col_stats = {
                "name": col,
                "type": col_type,
                "dtype": dtype,
                "missing_count": missing_count,
                "missing_percentage": round((missing_count / num_rows) * 100, 2) if num_rows > 0 else 0,
                "unique_count": unique_count
            }
            columns_info.append(col_stats)

        # Compile final profile report
        profile_report = {
            "num_rows": num_rows,
            "num_cols": num_cols,
            "duplicate_count": duplicate_count,
            "total_missing_values": missing_total,
            "columns": columns_info
        }

        return profile_report

    except Exception as e:
        print(f"Error during profiling: {e}")
        return {"error": str(e)}

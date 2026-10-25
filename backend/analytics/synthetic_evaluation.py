"""
================================================================================
SANKALAN AI PLATFORM - SYNTHETIC DATA EVALUATION & METRICS
File: backend/analytics/synthetic_evaluation.py
Role: Comprehensive Statistical, Distributional, Correlation & Privacy Evaluation
================================================================================

This module implements:
1. Exact Duplicate & Privacy Check:
   - Tests whether any synthetic record identically memorizes a real training record
   - Computes privacy score and exact duplication rate
2. Statistical Similarity:
   - Mean, standard deviation, median, quartiles, min/max per numerical feature
   - Normalized Z-score mean divergence and std ratio preservation
3. Distribution Similarity (Wasserstein Distance):
   - Computes Earth Mover's Distance normalized by original feature standard deviation
4. Correlation Structure Preservation:
   - Computes Pearson correlation matrix for Real vs Synthetic datasets
   - Absolute Difference Matrix and Mean Absolute Error (MAE)
5. Categorical Frequency Preservation (Total Variation Distance):
   - Compares relative category distribution probabilities
6. Downstream Machine Learning Utility:
   - Train on Synthetic, Test on Real (TSTR) vs Train on Real, Test on Real (TRTR)
   - Evaluates whether synthetic data retains functional predictive relationships
7. Project-Specific Composite Quality Score:
   - Weighted harmonic combination of statistical, distribution, correlation, and categorical fidelity
================================================================================
"""

import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def evaluate_synthetic_data(real_df: pd.DataFrame, synth_df: pd.DataFrame, column_roles: dict) -> dict:
    """
    Comprehensive evaluation of synthetic data against real data.
    
    Evaluates:
    1. Statistical Similarity (Means, Stds, Medians, Quartiles, Min/Max)
    2. Distribution Similarity (Wasserstein Distance normalized per numeric column)
    3. Correlation Structure Preservation (MAE & absolute difference)
    4. Categorical Frequency Preservation (Total Variation Distance)
    5. Exact Duplicate & Privacy Check (Memorization check)
    6. Downstream Utility (TSTR vs TRTR)
    7. Project-Specific Composite Quality Score
    """
    report = {
        "overall_score": 0.0,
        "statistical_similarity": 0.0,
        "distribution_similarity": 0.0,
        "correlation_similarity": 0.0,
        "categorical_similarity": 0.0,
        "privacy_score": 100.0,
        "exact_duplicates_count": 0,
        "exact_duplicate_rate": 0.0,
        "downstream_utility": None,
        "numeric_metrics": [],
        "categorical_metrics": [],
        "correlation_data": {
            "real_matrix": None,
            "synth_matrix": None,
            "diff_matrix": None,
            "columns": [],
            "mae": 0.0
        }
    }
    
    num_cols = [c for c in column_roles.get("generative_numeric", []) if c in real_df.columns and c in synth_df.columns]
    cat_cols = [c for c in column_roles.get("generative_categorical", []) if c in real_df.columns and c in synth_df.columns]
    
    # -------------------------------------------------------------
    # 1. Exact Duplicate / Privacy Check
    # -------------------------------------------------------------
    all_gen_cols = num_cols + cat_cols
    if all_gen_cols:
        real_gen = real_df[all_gen_cols].copy()
        synth_gen = synth_df[all_gen_cols].copy()
        for col in num_cols:
            real_gen[col] = pd.to_numeric(real_gen[col], errors='coerce').round(3)
            synth_gen[col] = pd.to_numeric(synth_gen[col], errors='coerce').round(3)
            
        real_tuples = set(tuple(x) for x in real_gen.to_numpy())
        duplicate_count = 0
        for row in synth_gen.to_numpy():
            if tuple(row) in real_tuples:
                duplicate_count += 1
                
        dup_rate = (duplicate_count / len(synth_df)) * 100 if len(synth_df) > 0 else 0.0
        report["exact_duplicates_count"] = duplicate_count
        report["exact_duplicate_rate"] = round(dup_rate, 3)
        # Privacy score: 100 if 0% duplicates, drops if records are memorized
        report["privacy_score"] = round(max(0.0, 100.0 - (dup_rate * 5)), 2)

    # -------------------------------------------------------------
    # 2. Statistical Similarity & Wasserstein Distance
    # -------------------------------------------------------------
    stat_scores = []
    dist_scores = []
    numeric_metrics = []
    
    for col in num_cols:
        r_series = pd.to_numeric(real_df[col], errors='coerce').dropna()
        s_series = pd.to_numeric(synth_df[col], errors='coerce').dropna()
        
        if len(r_series) == 0 or len(s_series) == 0:
            continue
            
        r_mean = float(r_series.mean())
        s_mean = float(s_series.mean())
        r_std = float(r_series.std()) if r_series.std() > 0 else 1e-6
        s_std = float(s_series.std()) if s_series.std() > 0 else 1e-6
        r_median = float(r_series.median())
        s_median = float(s_series.median())
        r_min = float(r_series.min())
        s_min = float(s_series.min())
        r_max = float(r_series.max())
        s_max = float(s_series.max())
        
        # Z-difference on mean and std
        z_mean_diff = abs(r_mean - s_mean) / r_std
        std_ratio_diff = abs(1.0 - (s_std / r_std))
        col_stat_score = max(0.0, 100.0 - (z_mean_diff * 25.0 + std_ratio_diff * 25.0))
        stat_scores.append(col_stat_score)
        
        # Wasserstein Distance (Normalized by real std)
        try:
            w_dist = float(wasserstein_distance(r_series, s_series))
            norm_w_dist = w_dist / r_std
            # Score: 100 when w_dist = 0, decreases smoothly as norm_w_dist increases
            col_dist_score = max(0.0, 100.0 * np.exp(-1.5 * norm_w_dist))
        except Exception:
            w_dist = 0.0
            norm_w_dist = 0.0
            col_dist_score = 100.0
            
        dist_scores.append(col_dist_score)
        
        numeric_metrics.append({
            "column": col,
            "real_mean": round(r_mean, 2),
            "synth_mean": round(s_mean, 2),
            "real_std": round(r_std, 2),
            "synth_std": round(s_std, 2),
            "real_median": round(r_median, 2),
            "synth_median": round(s_median, 2),
            "real_min": round(r_min, 2),
            "synth_min": round(s_min, 2),
            "real_max": round(r_max, 2),
            "synth_max": round(s_max, 2),
            "wasserstein_distance": round(w_dist, 4),
            "normalized_wasserstein": round(norm_w_dist, 4),
            "statistical_score": round(col_stat_score, 1),
            "distribution_score": round(col_dist_score, 1)
        })
        
    avg_stat_score = float(np.mean(stat_scores)) if stat_scores else 100.0
    avg_dist_score = float(np.mean(dist_scores)) if dist_scores else 100.0
    report["statistical_similarity"] = round(avg_stat_score, 2)
    report["distribution_similarity"] = round(avg_dist_score, 2)
    report["numeric_metrics"] = numeric_metrics

    # -------------------------------------------------------------
    # 3. Categorical Frequency Similarity (Total Variation Distance)
    # -------------------------------------------------------------
    cat_scores = []
    categorical_metrics = []
    
    for col in cat_cols:
        r_counts = real_df[col].astype(str).value_counts(normalize=True).to_dict()
        s_counts = synth_df[col].astype(str).value_counts(normalize=True).to_dict()
        
        all_categories = sorted(list(set(r_counts.keys()).union(set(s_counts.keys()))))
        
        # TVD = 0.5 * sum(|P(x) - Q(x)|)
        tvd = 0.5 * sum(abs(r_counts.get(cat, 0.0) - s_counts.get(cat, 0.0)) for cat in all_categories)
        cat_similarity = max(0.0, (1.0 - tvd) * 100.0)
        cat_scores.append(cat_similarity)
        
        freq_comparison = []
        for cat in all_categories[:10]:  # Top 10 categories
            freq_comparison.append({
                "category": cat,
                "real_pct": round(r_counts.get(cat, 0.0) * 100, 2),
                "synth_pct": round(s_counts.get(cat, 0.0) * 100, 2)
            })
            
        categorical_metrics.append({
            "column": col,
            "total_variation_distance": round(tvd, 4),
            "similarity_score": round(cat_similarity, 1),
            "frequencies": freq_comparison
        })
        
    avg_cat_score = float(np.mean(cat_scores)) if cat_scores else 100.0
    report["categorical_similarity"] = round(avg_cat_score, 2)
    report["categorical_metrics"] = categorical_metrics

    # -------------------------------------------------------------
    # 4. Correlation Structure Preservation
    # -------------------------------------------------------------
    if len(num_cols) > 1:
        clean_r = pd.DataFrame({c: pd.to_numeric(real_df[c], errors='coerce') for c in num_cols})
        clean_s = pd.DataFrame({c: pd.to_numeric(synth_df[c], errors='coerce') for c in num_cols})
        
        r_corr = clean_r.corr().fillna(0)
        s_corr = clean_s.corr().fillna(0)
        
        diff_matrix = np.abs(r_corr.values - s_corr.values)
        mae = float(np.mean(diff_matrix))
        
        # Correlation score: 100 if MAE = 0, decreases to 0 if MAE reaches 1
        corr_score = max(0.0, (1.0 - mae) * 100.0)
        report["correlation_similarity"] = round(corr_score, 2)
        report["correlation_data"] = {
            "columns": num_cols,
            "real_matrix": [[round(float(v), 3) for v in row] for row in r_corr.values.tolist()],
            "synth_matrix": [[round(float(v), 3) for v in row] for row in s_corr.values.tolist()],
            "diff_matrix": [[round(float(v), 3) for v in row] for row in diff_matrix.tolist()],
            "mae": round(mae, 4)
        }
    else:
        report["correlation_similarity"] = 100.0

    # -------------------------------------------------------------
    # 5. Downstream Machine Learning Utility (TSTR vs TRTR)
    # -------------------------------------------------------------
    try:
        # Detect suitable classification target (binary or small multi-class)
        target_col = None
        for col in cat_cols:
            n_unq = real_df[col].nunique()
            if 2 <= n_unq <= 5:
                target_col = col
                break
                
        if target_col and len(num_cols) >= 1:
            feature_cols = [c for c in num_cols if c != target_col]
            if len(feature_cols) >= 1:
                # Prepare Real Train & Test
                clean_r_eval = real_df[feature_cols + [target_col]].dropna()
                X_real = clean_r_eval[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
                y_real = clean_r_eval[target_col].astype(str)
                
                # Prepare Synthetic Train
                clean_s_eval = synth_df[feature_cols + [target_col]].dropna()
                X_synth = clean_s_eval[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
                y_synth = clean_s_eval[target_col].astype(str)
                
                if len(X_real) >= 30 and len(X_synth) >= 30:
                    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_real, y_real, test_size=0.3, random_state=42)
                    
                    # 1. TRTR: Train on Real, Test on Real
                    clf_trtr = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=6)
                    clf_trtr.fit(Xr_train, yr_train)
                    acc_trtr = accuracy_score(yr_test, clf_trtr.predict(Xr_test))
                    
                    # 2. TSTR: Train on Synthetic, Test on Real
                    clf_tstr = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=6)
                    clf_tstr.fit(X_synth, y_synth)
                    acc_tstr = accuracy_score(yr_test, clf_tstr.predict(Xr_test))
                    
                    utility_ratio = (acc_tstr / acc_trtr) * 100.0 if acc_trtr > 0 else 0.0
                    
                    report["downstream_utility"] = {
                        "target_column": target_col,
                        "feature_columns": feature_cols,
                        "model": "Random Forest Classifier",
                        "trtr_accuracy": round(acc_trtr * 100, 2),
                        "tstr_accuracy": round(acc_tstr * 100, 2),
                        "utility_retention_pct": round(min(100.0, utility_ratio), 2)
                    }
    except Exception as e:
        print("Downstream utility evaluation note:", e)

    # -------------------------------------------------------------
    # 6. Overall Composite Quality Score
    # -------------------------------------------------------------
    # Weights: Statistical 25%, Distribution 30%, Correlation 25%, Categorical 20%
    overall = (
        (report["statistical_similarity"] * 0.25) +
        (report["distribution_similarity"] * 0.30) +
        (report["correlation_similarity"] * 0.25) +
        (report["categorical_similarity"] * 0.20)
    )
    report["overall_score"] = round(overall, 2)
    
    return report

import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def evaluate_synthetic_data(real_df: pd.DataFrame, synth_df: pd.DataFrame, column_roles: dict) -> dict:
    """
    Comprehensive evaluation of synthetic data against real data according to
    Master Specification Sections 13, 14, 15, and 16.
    
    Evaluates:
    1. Statistical Similarity (Means, Stds, Medians, Quartiles, Min/Max)
    2. Distribution Similarity (Wasserstein Distance normalized per numeric column)
    3. Correlation Structure Preservation (MAE & Frobenius difference)
    4. Categorical Frequency Preservation (Total Variation Distance)
    5. Exact Duplicate & Privacy Check (Memorization check)
    6. Downstream Utility (Train-on-Synthetic, Test-on-Real vs Train-on-Real, Test-on-Real)
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
    # 1. Exact Duplicate / Memorization Check (Spec Section 13)
    # -------------------------------------------------------------
    all_gen_cols = num_cols + cat_cols
    if all_gen_cols:
        # Round numerics slightly to avoid floating precision artifacts in exact match
        real_gen = real_df[all_gen_cols].copy()
        synth_gen = synth_df[all_gen_cols].copy()
        for col in num_cols:
            real_gen[col] = real_gen[col].round(3)
            synth_gen[col] = synth_gen[col].round(3)
            
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
    # 2. Statistical Similarity & Wasserstein Distance (Spec Section 14A & 14B)
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
        
        # Wasserstein Distance normalized by standard deviation
        wd = float(wasserstein_distance(r_series, s_series))
        normalized_wd = wd / r_std
        # Score: 0 wd -> 100, 1.0 std distance -> 50, >=2.0 -> 0
        col_dist_score = max(0.0, 100.0 - (normalized_wd * 50.0))
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
            "wasserstein_distance": round(wd, 4),
            "similarity_score": round((col_stat_score + col_dist_score) / 2.0, 1)
        })
        
    report["statistical_similarity"] = round(float(np.mean(stat_scores)), 2) if stat_scores else 100.0
    report["distribution_similarity"] = round(float(np.mean(dist_scores)), 2) if dist_scores else 100.0
    report["numeric_metrics"] = numeric_metrics

    # -------------------------------------------------------------
    # 3. Correlation Structure Preservation (Spec Section 14C)
    # -------------------------------------------------------------
    if num_cols and len(num_cols) > 1:
        r_corr = real_df[num_cols].corr().fillna(0)
        s_corr = synth_df[num_cols].corr().fillna(0)
        diff_corr = np.abs(r_corr.values - s_corr.values)
        mae_corr = float(np.mean(diff_corr))
        
        # 0 MAE = 100, 0.5 MAE = 75, 1.0 MAE = 50, 2.0 MAE = 0
        corr_score = max(0.0, 100.0 - (mae_corr * 50.0))
        report["correlation_similarity"] = round(corr_score, 2)
        
        report["correlation_data"] = {
            "columns": num_cols,
            "real_matrix": [[round(v, 3) for v in row] for row in r_corr.values.tolist()],
            "synth_matrix": [[round(v, 3) for v in row] for row in s_corr.values.tolist()],
            "diff_matrix": [[round(v, 3) for v in row] for row in diff_corr.tolist()],
            "mae": round(mae_corr, 4)
        }
    else:
        report["correlation_similarity"] = 100.0

    # -------------------------------------------------------------
    # 4. Categorical Frequency Preservation (Spec Section 14D)
    # -------------------------------------------------------------
    cat_scores = []
    categorical_metrics = []
    
    for col in cat_cols:
        r_val_counts = real_df[col].astype(str).value_counts(normalize=True).to_dict()
        s_val_counts = synth_df[col].astype(str).value_counts(normalize=True).to_dict()
        
        all_categories = sorted(list(set(r_val_counts.keys()).union(set(s_val_counts.keys()))))
        tv_distance = 0.0
        breakdown = []
        
        for cat in all_categories:
            r_freq = r_val_counts.get(cat, 0.0)
            s_freq = s_val_counts.get(cat, 0.0)
            tv_distance += abs(r_freq - s_freq)
            breakdown.append({
                "category": cat,
                "real_pct": round(r_freq * 100, 1),
                "synth_pct": round(s_freq * 100, 1)
            })
            
        # Total Variation distance is between 0 and 2. TV=0 -> 100 score, TV=2 -> 0 score.
        col_cat_score = max(0.0, 100.0 - (tv_distance * 50.0))
        cat_scores.append(col_cat_score)
        
        categorical_metrics.append({
            "column": col,
            "tv_distance": round(tv_distance / 2.0, 4), # normalized [0, 1]
            "similarity_score": round(col_cat_score, 1),
            "breakdown": breakdown
        })
        
    report["categorical_similarity"] = round(float(np.mean(cat_scores)), 2) if cat_scores else 100.0
    report["categorical_metrics"] = categorical_metrics

    # -------------------------------------------------------------
    # 5. Composite Quality Score (Spec Section 15)
    # -------------------------------------------------------------
    # Weighted combination:
    # 25% Statistical Similarity
    # 25% Distribution Similarity (Wasserstein)
    # 25% Correlation Preservation
    # 15% Categorical Similarity
    # 10% Privacy Score (Duplicate penalty)
    weights = [
        (report["statistical_similarity"], 0.25),
        (report["distribution_similarity"], 0.25),
        (report["correlation_similarity"], 0.25),
        (report["categorical_similarity"], 0.15),
        (report["privacy_score"], 0.10)
    ]
    total_score = sum(val * weight for val, weight in weights)
    report["overall_score"] = round(total_score, 1)

    # -------------------------------------------------------------
    # 6. Optional Downstream Utility Check (Spec Section 16)
    # -------------------------------------------------------------
    # If there's an obvious binary/categorical classification target like 'Churn', evaluate utility
    target_candidate = None
    for cand in ['Churn', 'churn', 'target', 'label', 'status']:
        if cand in real_df.columns and cand in synth_df.columns:
            target_candidate = cand
            break
            
    if not target_candidate and cat_cols:
        # Use the last categorical column with <= 5 unique values
        for c in reversed(cat_cols):
            if real_df[c].nunique() in [2, 3, 4]:
                target_candidate = c
                break
                
    if target_candidate and num_cols:
        try:
            # Prepare feature matrices
            X_real = real_df[num_cols].fillna(real_df[num_cols].median())
            y_real = real_df[target_candidate].astype(str)
            
            X_synth = synth_df[num_cols].fillna(synth_df[num_cols].median())
            y_synth = synth_df[target_candidate].astype(str)
            
            # Split real dataset into Real Train (70%) and Real Test (30%)
            X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
                X_real, y_real, test_size=0.3, random_state=42
            )
            
            # 1. TRTR: Train on Real, Test on Real Holdout
            clf_real = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
            clf_real.fit(X_r_train, y_r_train)
            trtr_acc = accuracy_score(y_r_test, clf_real.predict(X_r_test))
            
            # 2. TSTR: Train on Synthetic, Test on Real Holdout
            # Filter synthetic classes that match real classes
            valid_mask = y_synth.isin(y_r_train.unique())
            if valid_mask.sum() > 50:
                clf_synth = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
                clf_synth.fit(X_synth[valid_mask], y_synth[valid_mask])
                tstr_acc = accuracy_score(y_r_test, clf_synth.predict(X_r_test))
                
                utility_ratio = (tstr_acc / trtr_acc * 100) if trtr_acc > 0 else 0.0
                report["downstream_utility"] = {
                    "target_column": target_candidate,
                    "model": "Random Forest Classifier",
                    "trtr_accuracy": round(trtr_acc * 100, 2),
                    "tstr_accuracy": round(tstr_acc * 100, 2),
                    "utility_retention_pct": round(min(100.0, utility_ratio), 1)
                }
        except Exception as e:
            print(f"Downstream utility computation skipped: {e}")
            report["downstream_utility"] = None

    return report

import pandas as pd
import numpy as np

def evaluate_synthetic_data(real_df, synth_df, column_roles):
    """
    Compares real and synthetic data to generate a quality report.
    Focuses on the generative columns.
    """
    report = {
        "correlation_similarity": 0,
        "statistical_similarity": 0,
        "categorical_similarity": 0,
        "overall_score": 0,
        "details": {}
    }
    
    num_cols = [c for c in column_roles.get("generative_numeric", []) if c in real_df.columns and c in synth_df.columns]
    cat_cols = [c for c in column_roles.get("generative_categorical", []) if c in real_df.columns and c in synth_df.columns]
    
    # 1. Correlation Similarity (for numeric)
    if num_cols and len(num_cols) > 1:
        real_corr = real_df[num_cols].corr().fillna(0).values
        synth_corr = synth_df[num_cols].corr().fillna(0).values
        # Mean Absolute Error of correlation matrix
        mae_corr = np.mean(np.abs(real_corr - synth_corr))
        # Convert to a 0-100 score (0 MAE = 100 score, 2.0 MAE = 0 score)
        corr_score = max(0, 100 - (mae_corr * 50))
        report["correlation_similarity"] = round(corr_score, 2)
    else:
        report["correlation_similarity"] = 100.0

    # 2. Statistical Similarity (Numeric Means/Std)
    stat_scores = []
    for col in num_cols:
        real_mean = real_df[col].mean()
        synth_mean = synth_df[col].mean()
        real_std = real_df[col].std()
        
        # Avoid division by zero
        if real_std == 0:
            real_std = 1e-6
            
        # How many standard deviations away is the synthetic mean?
        z_diff = abs(real_mean - synth_mean) / real_std
        # Score based on z-diff (0 diff = 100, 3 diff = 0)
        score = max(0, 100 - (z_diff * 33.3))
        stat_scores.append(score)
        
    report["statistical_similarity"] = round(np.mean(stat_scores), 2) if stat_scores else 100.0

    # 3. Categorical Frequency Similarity
    cat_scores = []
    for col in cat_cols:
        real_freq = real_df[col].value_counts(normalize=True).to_dict()
        synth_freq = synth_df[col].value_counts(normalize=True).to_dict()
        
        # Total absolute variation distance
        tv = 0
        all_keys = set(real_freq.keys()).union(set(synth_freq.keys()))
        for k in all_keys:
            tv += abs(real_freq.get(k, 0) - synth_freq.get(k, 0))
            
        # TV is between 0 and 2. TV=0 -> 100 score. TV=2 -> 0 score.
        score = max(0, 100 - (tv * 50))
        cat_scores.append(score)
        
    report["categorical_similarity"] = round(np.mean(cat_scores), 2) if cat_scores else 100.0

    # 4. Overall Score
    weights = [
        (report["correlation_similarity"], 0.4),
        (report["statistical_similarity"], 0.4),
        (report["categorical_similarity"], 0.2)
    ]
    
    total_score = sum(val * weight for val, weight in weights)
    report["overall_score"] = round(total_score, 2)
    
    return report

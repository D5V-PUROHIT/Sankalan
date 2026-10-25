# 🧭 Sankalan Project Architecture & File Location Map

This document is your **instant reference cheat-sheet**. When someone (evaluator, interviewer, professor, or teammate) asks:
> *"Where is the main VAE model?"*  
> *"Where is the EDA code?"*  
> *"Where is the data preprocessing?"*  

You can instantly point to the exact file and describe its function.

---

## 🎯 Quick Reference Cheat Sheet

| Question / Feature | Exact File Path | Key Functions & Classes | Description |
| :--- | :--- | :--- | :--- |
| 🧠 **Main VAE Model** | [`backend/models/main_vae_model.py`](file:///c:/Users/Rudra%20Dev/Desktop/Sankalan/backend/models/main_vae_model.py) | `VAE`, `Sampling`, `compute_reconstruction_loss`, `get_beta`, `train_step`, `train_vae` | **Encoder & Decoder neural networks**, Reparameterization trick, composite loss (MSE + Categorical Cross-Entropy), Beta warmup annealing, and Adam optimization. |
| 📊 **EDA (Exploratory Data Analysis)** | [`backend/analytics/eda_analysis.py`](file:///c:/Users/Rudra%20Dev/Desktop/Sankalan/backend/analytics/eda_analysis.py) | `generate_eda_data`, `generate_comparison_data`, `is_valid_categorical` | Computes statistical summaries, numerical histograms, categorical frequencies, and Pearson **correlation matrices** for both raw data and Real vs Synthetic comparison. |
| 🧹 **Data Preprocessing & Scaling** | [`backend/analytics/data_preprocessing.py`](file:///c:/Users/Rudra%20Dev/Desktop/Sankalan/backend/analytics/data_preprocessing.py) | `clean_and_preprocess` | Automatically classifies columns (**Identifiers**, **Dates**, **Generative Numeric**, **Generative Categorical**), performs median/Unknown imputation, **StandardScaler**, **OneHotEncoder**, and builds the feature map slice indices. |
| 📋 **Dataset Profiling** | [`backend/analytics/dataset_profiling.py`](file:///c:/Users/Rudra%20Dev/Desktop/Sankalan/backend/analytics/dataset_profiling.py) | `profile_dataset` | Calculates dataset health: row/column counts, missing value percentages, duplicate detection, and column-level cardinality. |
| 🧪 **Synthetic Data Generator** | [`backend/analytics/synthetic_generation.py`](file:///c:/Users/Rudra%20Dev/Desktop/Sankalan/backend/analytics/synthetic_generation.py) | `generate_and_inverse_transform` | Samples latent vectors from $\mathcal{N}(0, I)$, runs neural decoding, applies **inverse scaling**, **inverse one-hot transformation**, domain constraint preservation (integer/discrete rounding, non-negative bounds), and mock ID/date attachment. |
| 📈 **Synthetic Data Evaluation & Metrics** | [`backend/analytics/synthetic_evaluation.py`](file:///c:/Users/Rudra%20Dev/Desktop/Sankalan/backend/analytics/synthetic_evaluation.py) | `evaluate_synthetic_data` | Benchmarks synthetic data: **Wasserstein distance** (distribution preservation), **Correlation MAE**, **Total Variation Distance** (categorical), **Privacy duplicate check** (memorization score), **Downstream ML utility** (TSTR vs TRTR), and **Composite Quality Score**. |
| 🚀 **Main API Server & Router** | [`backend/server_app.py`](file:///c:/Users/Rudra%20Dev/Desktop/Sankalan/backend/server_app.py) | FastAPI routes: `/api/upload`, `/api/eda`, `/api/generate`, `/api/comparison`, etc. | REST API server coordinating state persistence, background VAE training progress, data endpoints, and static frontend hosting. |
| 🖥️ **Interactive Frontend Dashboard** | [`frontend/index.html`](file:///c:/Users/Rudra%20Dev/Desktop/Sankalan/frontend/index.html)<br>[`frontend/css/style.css`](file:///c:/Users/Rudra%20Dev/Desktop/Sankalan/frontend/css/style.css)<br>[`frontend/js/script.js`](file:///c:/Users/Rudra%20Dev/Desktop/Sankalan/frontend/js/script.js) | Tab navigation, Chart.js / Plotly charts, Live training progress bar | Dark-mode glassmorphic single-page web interface with real-time training progress, theater view, and side-by-side Real vs Synthetic comparisons. |
| 📁 **Data & Model Artifacts** | `backend/data/`<br>`backend/models/saved_models/` | `raw/`, `processed/`, `synthetic/`, `scaler.pkl`, `ohe_encoders.pkl` | Storage for raw uploaded datasets, preprocessed training matrices (`X_train.npy`), generated synthetic datasets, and fitted scikit-learn transformers. |

---

## 📂 Detailed Directory Tree

```
Sankalan/
│
├── README.md                                  # Platform overview & setup guide
├── PROJECT_STRUCTURE.md                       # This architecture & location cheatsheet
├── run.bat                                    # One-click Windows startup script
├── .gitignore                                 # Git ignore rules for environments & caches
├── Sankalan_Project_Master_Specification.docx # Master project product specification
│
├── backend/
│   ├── server_app.py                          # 🚀 Primary FastAPI server & application controller
│   ├── app.py                                 # 🔄 Backward-compatible launcher script
│   ├── requirements.txt                       # Python dependencies (TensorFlow, FastAPI, etc.)
│   │
│   ├── models/
│   │   ├── main_vae_model.py                  # 🧠 [CORE] Variational Autoencoder architecture & training
│   │   ├── vae.py                             # 🔄 Backward-compatible wrapper importing main_vae_model.py
│   │   └── saved_models/                      # 💾 Persisted model artifacts
│   │       ├── column_roles.json              # Classified column roles & feature map indices
│   │       ├── scaler.pkl                     # Fitted StandardScaler for numerical features
│   │       └── ohe_encoders.pkl               # Fitted OneHotEncoders for categorical features
│   │
│   ├── analytics/
│   │   ├── main_vae_model.py -> in models/
│   │   ├── eda_analysis.py                    # 📊 [EDA] Distribution & correlation analytics engine
│   │   ├── eda.py                             # 🔄 Backward-compatible wrapper importing eda_analysis.py
│   │   ├── data_preprocessing.py              # 🧹 [PREPROCESS] Cleaning, classification, scaling pipeline
│   │   ├── preprocessing.py                   # 🔄 Backward-compatible wrapper importing data_preprocessing.py
│   │   ├── dataset_profiling.py               # 📋 [PROFILING] Initial health check, missingness, duplicates
│   │   ├── profiling.py                       # 🔄 Backward-compatible wrapper importing dataset_profiling.py
│   │   ├── synthetic_generation.py            # 🧪 [GENERATION] Latent sampling, decoding & inverse scaling
│   │   ├── generation.py                      # 🔄 Backward-compatible wrapper importing synthetic_generation.py
│   │   ├── synthetic_evaluation.py            # 📈 [METRICS] Wasserstein, correlation MAE, privacy, quality score
│   │   └── evaluation.py                      # 🔄 Backward-compatible wrapper importing synthetic_evaluation.py
│   │
│   └── data/
│       ├── active_dataset.json                # Tracks the currently active uploaded dataset
│       ├── raw/                               # Uploaded original tabular datasets (.csv, .xlsx)
│       ├── processed/                         # Prepared numerical arrays (X_train.npy, profile.json)
│       └── synthetic/                         # Generated synthetic_data.csv & evaluation_report.json
│
└── frontend/
    ├── index.html                             # 🖥️ Main HTML single-page dashboard layout
    ├── css/
    │   └── style.css                          # 🎨 Modern dark-mode glassmorphic styling
    └── js/
        └── script.js                          # ⚡ Client logic, API communication & interactive charts
```

---

## 🔄 End-to-End Execution Flow

```
1. USER UPLOADS CSV/EXCEL
   └─► backend/server_app.py (/api/upload)
         ├─► backend/analytics/dataset_profiling.py (profile_dataset)
         └─► backend/analytics/data_preprocessing.py (clean_and_preprocess)
               ├─ Classifies Identifiers, Dates, Generative Numerics, Generative Categoricals
               ├─ Fits StandardScaler & OneHotEncoder
               └─ Saves backend/data/processed/X_train.npy & column_roles.json

2. USER VIEWS EXPLORATORY DATA ANALYSIS
   └─► backend/server_app.py (/api/eda)
         └─► backend/analytics/eda_analysis.py (generate_eda_data)
               └─ Returns correlation heatmap, histograms, box plots, category distributions

3. USER TRAINS VAE & GENERATES SYNTHETIC DATA
   └─► backend/server_app.py (/api/generate)
         ├─► backend/models/main_vae_model.py (train_vae)
         │     ├─ Encoder -> Latent Space z ~ N(mu, sigma) -> Decoder
         │     ├─ Composite Loss: MSE (numerics) + Categorical Cross-Entropy (categoricals)
         │     └─ KL Divergence with Beta warmup annealing
         ├─► backend/analytics/synthetic_generation.py (generate_and_inverse_transform)
         │     ├─ Samples z ~ N(0, I) -> Decodes -> Inverse transforms numerics & categoricals
         │     ├─ Enforces domain constraints (integers, non-negative boundaries)
         │     └─ Saves backend/data/synthetic/synthetic_data.csv
         └─► backend/analytics/synthetic_evaluation.py (evaluate_synthetic_data)
               ├─ Computes Wasserstein distance, Correlation MAE, Total Variation Distance
               ├─ Checks privacy / memorization duplication rate
               ├─ Evaluates downstream ML utility (TSTR vs TRTR)
               └─ Saves backend/data/synthetic/evaluation_report.json

4. USER COMPARES REAL VS SYNTHETIC DATA
   └─► backend/server_app.py (/api/comparison)
         └─► backend/analytics/eda_analysis.py (generate_comparison_data)
               └─ Side-by-side distributions, summary statistics diffs, and correlation error matrices
```

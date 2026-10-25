# Sankalan (संकलन) ⚡
### Deep Learning-Based Tabular Synthetic Data Generation & Analytics Platform

Sankalan is an end-to-end Deep Learning platform that learns complex structural patterns and probability distributions from tabular datasets (CSV and Excel) using a **Variational Autoencoder (VAE)**, generates realistic synthetic records, and benchmarks them using rigorous statistical, distributional, correlation, and privacy metrics.

---

## 🧭 Component Quick Navigation

Need to locate specific code for presentations or reviews? Check [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) or use this direct map:

| Component | File Location | Key Purpose |
| :--- | :--- | :--- |
| 🧠 **Main VAE Deep Learning Model** | [`backend/models/main_vae_model.py`](backend/models/main_vae_model.py) | Encoder, Latent Sampling, Decoder, Composite Loss (MSE + Categorical Cross-Entropy), Beta Annealing & Custom Training Loop. |
| 📊 **Exploratory Data Analysis (EDA)** | [`backend/analytics/eda_analysis.py`](backend/analytics/eda_analysis.py) | Pearson correlation heatmaps, numerical distribution histograms, box plots, and category frequency counts. |
| 🧹 **Data Preprocessing & Scaling** | [`backend/analytics/data_preprocessing.py`](backend/analytics/data_preprocessing.py) | Automatic column role classification, missing value imputation, `StandardScaler`, `OneHotEncoder`, and feature mapping. |
| 📋 **Dataset Profiling** | [`backend/analytics/dataset_profiling.py`](backend/analytics/dataset_profiling.py) | Automated dataset health check: row/column counts, missing percentages, and duplicate detection. |
| 🧪 **Synthetic Data Generator** | [`backend/analytics/synthetic_generation.py`](backend/analytics/synthetic_generation.py) | Latent space sampling $z \sim \mathcal{N}(0, I)$, inverse scaling, inverse one-hot transformation, domain constraints, and mock IDs/dates. |
| 📈 **Synthetic Data Evaluation** | [`backend/analytics/synthetic_evaluation.py`](backend/analytics/synthetic_evaluation.py) | Wasserstein distance, Correlation MAE, Total Variation Distance, duplicate privacy check, and downstream ML utility. |
| 🚀 **API Server & Controller** | [`backend/server_app.py`](backend/server_app.py) | FastAPI backend serving REST endpoints (`/upload`, `/eda`, `/train`, `/generate`, `/comparison`, etc.) and frontend assets. |
| 🖥️ **Interactive Dashboard** | [`frontend/index.html`](frontend/index.html) | Glassmorphic, dark-mode single-page web dashboard with live training progress bars and side-by-side comparison. |

---

## 🌟 Key Features

1. **Intelligent Tabular Preprocessing**:
   - Accepts both `.csv` and Excel (`.xlsx`, `.xls`) files.
   - Automatically separates features into **Identifiers**, **Dates**, **Generative Numeric**, and **Generative Categorical**.
   - Handles missing values via median imputation (numerics) and explicit class imputation (categoricals).

2. **Custom Variational Autoencoder Architecture**:
   - **Encoder**: Maps multidimensional feature vectors into a Gaussian latent distribution parameterized by mean ($\mu$) and log variance ($\log \sigma^2$).
   - **Reparameterization Trick**: Enables backpropagation through stochastic sampling: $z = \mu + \sigma \odot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$.
   - **Decoder**: Reconstructs continuous numeric features and one-hot categorical logits.
   - **Composite Loss Function**:
     $$\mathcal{L}_{\text{total}} = \text{MSE}(\text{numeric}) + \text{CCE}(\text{categorical}) + \beta \cdot D_{\text{KL}}(q(z|x) \,||\, p(z))$$
   - **Beta Annealing**: Warmup scheduler gradually ramps up KL Divergence regularization to prevent posterior collapse.

3. **Domain Constraint Preservation**:
   - Enforces non-negative boundaries on strictly non-negative real-world attributes (e.g., prices, charges, age).
   - Rounds and clips discrete and integer columns.
   - Resamples empirical dates and generates realistic mock identifiers (`SYN-XXXXXX`).

4. **Rigorous Quality & Privacy Benchmarking**:
   - **Wasserstein Distance**: Measures Earth Mover's Distance between real and synthetic continuous distributions.
   - **Correlation Matrix MAE**: Measures preservation of inter-feature relationships.
   - **Total Variation Distance**: Evaluates categorical frequency preservation.
   - **Privacy Duplicate Check**: Verifies that the model has not memorized training records (0% duplication rate).
   - **Downstream Utility (TSTR vs TRTR)**: Evaluates machine learning utility by comparing Train-on-Synthetic, Test-on-Real against Train-on-Real, Test-on-Real.

---

## 🚀 Quick Start

### 1. Requirements
- Python 3.10 or 3.11 (recommended)
- Windows / macOS / Linux

### 2. Installation
Install dependencies:
```bash
pip install -r backend/requirements.txt
```

### 3. Launching the Platform

**On Windows:**  
Double-click `run.bat` or run:
```cmd
run.bat
```

**Using Python Directly:**
```bash
python backend/server_app.py
```
*(Or `python backend/app.py`)*

Then open your browser and navigate to:  
👉 **http://127.0.0.1:8000**

---

## 📊 Demonstration Workflow

1. **Upload Dataset**: Upload any tabular CSV or Excel file (e.g. Telco Churn, Penguins, or Hockey dataset).
2. **Review Profiling & Health**: Inspect row/column counts, missing percentages, and duplicate metrics.
3. **Explore Data (EDA)**: View correlation heatmaps, distribution histograms, and category frequency counts.
4. **Train VAE**: Set the desired synthetic record count and training epochs; observe live loss curves and progress bars.
5. **Evaluate & Compare**: Inspect Wasserstein distances, correlation difference matrices, privacy scores, and side-by-side distributions.
6. **Export**: Download the synthetic dataset (`.csv`) and the evaluation benchmark report (`.json`).

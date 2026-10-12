from fastapi import FastAPI, UploadFile, File, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import os
import uvicorn
import glob
import json
import pandas as pd
import numpy as np

from analytics.profiling import profile_dataset
from analytics.preprocessing import clean_and_preprocess
from analytics.eda import generate_eda_data, generate_comparison_data

app = FastAPI(title="Sankalan API", description="Deep Learning Synthetic Data & Analytics Platform")

# Define absolute paths
BASE_DIR = os.path.dirname(__file__)
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), 'frontend')
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
SYNTHETIC_DIR = os.path.join(BASE_DIR, 'data', 'synthetic')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def get_latest_raw_file():
    raw_files = glob.glob(os.path.join(RAW_DIR, '*'))
    return raw_files[0] if raw_files else None


def get_synthetic_file():
    synth_path = os.path.join(SYNTHETIC_DIR, 'synthetic_data.csv')
    return synth_path if os.path.exists(synth_path) else None


@app.get("/api/status")
async def get_system_status():
    """Returns the current state of data and models in the system."""
    raw_file = get_latest_raw_file()
    synth_file = get_synthetic_file()
    report_file = os.path.join(SYNTHETIC_DIR, 'evaluation_report.json')
    
    status = {
        "dataset_loaded": raw_file is not None,
        "raw_filename": os.path.basename(raw_file) if raw_file else None,
        "synthetic_ready": synth_file is not None,
        "evaluation_ready": os.path.exists(report_file)
    }
    
    if synth_file and os.path.exists(report_file):
        try:
            with open(report_file, 'r') as f:
                report_data = json.load(f)
            status["latest_quality_score"] = report_data.get("overall_score", None)
        except Exception:
            pass
            
    return status


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Receives a dataset, cleans it, profiles it, and prepares VAE preprocessing tensors.
    """
    if not file.filename.endswith(('.csv', '.xlsx')):
        return JSONResponse(status_code=400, content={"message": "Invalid file format. Only CSV and Excel supported."})
        
    file_path = os.path.join(RAW_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
        
    # Profile dataset
    profile_result = profile_dataset(file_path)
    if "error" in profile_result:
        return JSONResponse(status_code=500, content={"message": f"Error profiling dataset: {profile_result['error']}"})
        
    # Preprocess dataset for VAE
    preprocess_result = clean_and_preprocess(
        file_path, 
        output_dir=PROCESSED_DIR,
        models_dir=MODELS_DIR
    )
        
    return {
        "message": "File uploaded, profiled, and preprocessed successfully", 
        "filename": file.filename, 
        "profile": profile_result,
        "preprocessing": preprocess_result
    }


@app.get("/api/eda")
async def get_eda_data():
    """
    Returns rich exploratory data analysis charts data for the raw dataset.
    """
    raw_file_path = get_latest_raw_file()
    if not raw_file_path:
        return JSONResponse(status_code=400, content={"message": "No raw dataset found. Please upload one first."})
        
    eda_data = generate_eda_data(raw_file_path)
    if eda_data.get("status") == "error":
        return JSONResponse(status_code=500, content={"message": f"EDA failed: {eda_data.get('message')}"})
        
    return eda_data


@app.post("/api/generate")
async def generate_synthetic_data(
    num_records: int = Query(5000, ge=50, le=100000), 
    epochs: int = Query(50, ge=5, le=500),
    latent_dim: int = Query(8, ge=2, le=64)
):
    """
    Trains the Variational Autoencoder, samples from latent space, decodes synthetic records,
    and runs the comprehensive evaluation suite.
    """
    from models.vae import train_vae
    from analytics.generation import generate_and_inverse_transform
    from analytics.evaluation import evaluate_synthetic_data

    raw_file_path = get_latest_raw_file()
    if not raw_file_path:
        return JSONResponse(status_code=400, content={"message": "No raw dataset found. Please upload one first."})
        
    x_train_path = os.path.join(PROCESSED_DIR, 'X_train.npy')
    roles_path = os.path.join(MODELS_DIR, 'column_roles.json')
    
    if not os.path.exists(x_train_path) or not os.path.exists(roles_path):
        return JSONResponse(status_code=400, content={"message": "Dataset not yet preprocessed. Please re-upload dataset."})

    try:
        X_train = np.load(x_train_path)
        with open(roles_path, 'r') as f:
            metadata = json.load(f)
            
        feature_map = metadata["feature_map"]
        column_roles = metadata["roles"]
        
        # 1. Train VAE
        vae_model, history = train_vae(X_train, feature_map, latent_dim=latent_dim, epochs=epochs, batch_size=32)
        
        # 2. Generate and inverse transform synthetic data
        synth_df = generate_and_inverse_transform(
            vae_model, feature_map, column_roles, num_records, MODELS_DIR, raw_file_path
        )
        
        # 3. Save synthetic CSV
        synth_path = os.path.join(SYNTHETIC_DIR, 'synthetic_data.csv')
        synth_df.to_csv(synth_path, index=False)
        
        # 4. Comprehensive Evaluation
        real_df = pd.read_csv(raw_file_path) if raw_file_path.endswith('.csv') else pd.read_excel(raw_file_path)
        eval_report = evaluate_synthetic_data(real_df, synth_df, column_roles)
        
        # Save evaluation report to JSON
        report_path = os.path.join(SYNTHETIC_DIR, 'evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(eval_report, f, indent=4)
            
        return {
            "message": "Generation and evaluation complete", 
            "records_generated": num_records,
            "training_history": history,
            "quality_report": eval_report,
            "download_url": "/api/download",
            "report_download_url": "/api/download/report"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"Generation failed: {str(e)}"})


@app.get("/api/comparison")
async def get_comparison_data(num_col: str = None, cat_col: str = None):
    """
    Returns comparative analytics, distributions, and correlation matrices
    between Real and Synthetic datasets.
    """
    raw_file_path = get_latest_raw_file()
    synth_file_path = get_synthetic_file()
    
    if not raw_file_path:
        return JSONResponse(status_code=400, content={"message": "No raw dataset found."})
    if not synth_file_path:
        return JSONResponse(status_code=400, content={"message": "No synthetic data generated yet. Please generate data first."})
        
    comparison_data = generate_comparison_data(raw_file_path, synth_file_path, num_col=num_col, cat_col=cat_col)
    if comparison_data.get("status") == "error":
        return JSONResponse(status_code=500, content={"message": f"Comparison failed: {comparison_data.get('message')}"})
        
    return comparison_data


@app.get("/api/download")
async def download_synthetic():
    """Serves the generated synthetic CSV file."""
    synth_path = get_synthetic_file()
    if synth_path:
        return FileResponse(synth_path, media_type='text/csv', filename="synthetic_dataset.csv")
    return JSONResponse(status_code=404, content={"message": "Synthetic dataset not found."})


@app.get("/api/download/report")
async def download_report():
    """Serves the evaluation report as a JSON file."""
    report_path = os.path.join(SYNTHETIC_DIR, 'evaluation_report.json')
    if os.path.exists(report_path):
        return FileResponse(report_path, media_type='application/json', filename="quality_evaluation_report.json")
    return JSONResponse(status_code=404, content={"message": "Evaluation report not found."})


# Mount frontend static assets
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    print("Starting Sankalan Backend on port 8000...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

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

# Global Live Training Progress State
TRAINING_STATE = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "loss": 0.0,
    "recon_loss": 0.0,
    "kl_loss": 0.0,
    "percent": 0.0,
    "status": "idle",
    "stop_requested": False,
    "message": "Ready"
}


ACTIVE_DATASET_META = os.path.join(BASE_DIR, 'data', 'active_dataset.json')

def get_latest_raw_file():
    # 1. Check if an active dataset was designated on upload
    if os.path.exists(ACTIVE_DATASET_META):
        try:
            with open(ACTIVE_DATASET_META, 'r') as f:
                meta = json.load(f)
                active_path = os.path.join(RAW_DIR, meta.get("filename", ""))
                if os.path.exists(active_path):
                    return active_path
        except Exception:
            pass

    # 2. Fallback to newest modified file in raw directory
    raw_files = glob.glob(os.path.join(RAW_DIR, '*'))
    if not raw_files:
        return None
    return max(raw_files, key=os.path.getmtime)


def get_synthetic_file():
    synth_path = os.path.join(SYNTHETIC_DIR, 'synthetic_data.csv')
    return synth_path if os.path.exists(synth_path) else None


@app.get("/api/status")
async def get_system_status():
    """Returns the full persistent state of data, training, and models in the system."""
    raw_file = get_latest_raw_file()
    synth_file = get_synthetic_file()
    report_file = os.path.join(SYNTHETIC_DIR, 'evaluation_report.json')
    history_file = os.path.join(SYNTHETIC_DIR, 'training_history.json')
    profile_cache = os.path.join(PROCESSED_DIR, 'profile.json')
    roles_cache = os.path.join(MODELS_DIR, 'column_roles.json')
    
    status = {
        "dataset_loaded": raw_file is not None,
        "raw_filename": os.path.basename(raw_file) if raw_file else None,
        "synthetic_ready": synth_file is not None,
        "evaluation_ready": os.path.exists(report_file),
        "training": TRAINING_STATE,
        "profile": None,
        "preprocessing": None,
        "evaluation_report": None,
        "training_history": None
    }
    
    # 1. Restore Profile Cache
    if raw_file:
        if os.path.exists(profile_cache):
            try:
                with open(profile_cache, 'r') as f:
                    status["profile"] = json.load(f)
            except Exception:
                pass
        else:
            profile_data = profile_dataset(raw_file)
            if "error" not in profile_data:
                try:
                    with open(profile_cache, 'w') as f:
                        json.dump(profile_data, f)
                    status["profile"] = profile_data
                except Exception:
                    pass

        # 2. Restore Preprocessing Schema Roles
        if os.path.exists(roles_cache):
            try:
                with open(roles_cache, 'r') as f:
                    roles_meta = json.load(f)
                    roles = roles_meta.get("roles", {})
                    status["preprocessing"] = {
                        "vae_input_dim": roles_meta.get("total_generative_dim", 0),
                        "roles_assigned": {
                            "identifiers": len(roles.get("identifiers", [])),
                            "dates": len(roles.get("dates", [])),
                            "numeric": len(roles.get("generative_numeric", [])),
                            "categorical": len(roles.get("generative_categorical", []))
                        }
                    }
            except Exception:
                pass

    # 3. Restore Evaluation Report (only if synth dataset matches current raw dataset)
    if synth_file and raw_file:
        try:
            r_df = pd.read_csv(raw_file, nrows=1) if raw_file.lower().endswith('.csv') else pd.read_excel(raw_file, nrows=1)
            s_df = pd.read_csv(synth_file, nrows=1)
            common = set(r_df.columns).intersection(set(s_df.columns))
            if len(common) < len(r_df.columns) * 0.4:
                synth_file = None
                status["synthetic_ready"] = False
        except Exception:
            pass

    if synth_file and os.path.exists(report_file):
        try:
            with open(report_file, 'r') as f:
                report_data = json.load(f)
                status["evaluation_report"] = report_data
                status["latest_quality_score"] = report_data.get("overall_score", None)
        except Exception:
            pass

    # 4. Restore Training History
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                status["training_history"] = json.load(f)
        except Exception:
            pass
            
    return status


@app.get("/api/train/status")
async def get_train_status():
    """Live polling endpoint for continuous progress updates and epoch monitoring."""
    return TRAINING_STATE


@app.post("/api/train/stop")
async def stop_training():
    """Signal the training loop to stop immediately at the next epoch."""
    if TRAINING_STATE["is_training"]:
        TRAINING_STATE["stop_requested"] = True
        TRAINING_STATE["message"] = "Stopping training..."
        return {"status": "success", "message": "Stop requested. Training will halt at the current epoch."}
    return {"status": "idle", "message": "No active training in progress."}


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Receives a dataset, cleans it, profiles it, and prepares VAE preprocessing tensors.
    """
    if not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
        return JSONResponse(status_code=400, content={"message": "Invalid file format. Only CSV and Excel (.xlsx, .xls) supported."})
        
    file_path = os.path.join(RAW_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
        
    # Profile dataset
    profile_result = profile_dataset(file_path)
    if "error" in profile_result:
        return JSONResponse(status_code=500, content={"message": f"Error profiling dataset: {profile_result['error']}"})
        
    # Cache profile dataset
    profile_cache = os.path.join(PROCESSED_DIR, 'profile.json')
    try:
        with open(profile_cache, 'w') as f:
            json.dump(profile_result, f)
    except Exception:
        pass
        
    # Preprocess dataset for VAE
    preprocess_result = clean_and_preprocess(
        file_path, 
        output_dir=PROCESSED_DIR,
        models_dir=MODELS_DIR
    )
        
    # Save active dataset tracking
    try:
        with open(ACTIVE_DATASET_META, 'w') as f:
            json.dump({"filename": file.filename}, f)
    except Exception:
        pass

    # Clear prior synthetic outputs for the previous dataset
    for fname in ['synthetic_data.csv', 'evaluation_report.json', 'training_history.json']:
        p = os.path.join(SYNTHETIC_DIR, fname)
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

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
    epochs: int = Query(40, ge=5, le=500),
    latent_dim: int = Query(8, ge=2, le=64)
):
    """
    Trains the Variational Autoencoder with live progress callbacks, samples from latent space,
    decodes synthetic records, and executes the evaluation suite.
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
        # Initialize training state
        TRAINING_STATE["is_training"] = True
        TRAINING_STATE["stop_requested"] = False
        TRAINING_STATE["status"] = "training"
        TRAINING_STATE["current_epoch"] = 0
        TRAINING_STATE["total_epochs"] = epochs
        TRAINING_STATE["percent"] = 5.0
        TRAINING_STATE["message"] = f"Initializing VAE architecture for {epochs} epochs..."

        X_train = np.load(x_train_path)
        with open(roles_path, 'r') as f:
            metadata = json.load(f)
            
        feature_map = metadata["feature_map"]
        column_roles = metadata["roles"]
        
        def on_epoch_progress(epoch, total_epochs, loss, recon, kl):
            TRAINING_STATE["current_epoch"] = epoch
            TRAINING_STATE["total_epochs"] = total_epochs
            TRAINING_STATE["loss"] = round(loss, 4)
            TRAINING_STATE["recon_loss"] = round(recon, 4)
            TRAINING_STATE["kl_loss"] = round(kl, 4)
            pct = 5.0 + (epoch / total_epochs) * 80.0
            TRAINING_STATE["percent"] = round(pct, 1)
            TRAINING_STATE["message"] = f"Epoch {epoch}/{total_epochs} • Loss: {loss:.3f} (Recon: {recon:.3f}, KL: {kl:.3f})"

        def check_stop():
            return TRAINING_STATE["stop_requested"]

        # 1. Train VAE
        vae_model, history = train_vae(
            X_train, 
            feature_map, 
            latent_dim=latent_dim, 
            epochs=epochs, 
            batch_size=32,
            progress_callback=on_epoch_progress,
            check_stop_callback=check_stop
        )
        
        TRAINING_STATE["percent"] = 88.0
        TRAINING_STATE["message"] = f"Synthesizing {num_records:,} records and applying inverse transformation..."

        # 2. Generate and inverse transform synthetic data
        synth_df = generate_and_inverse_transform(
            vae_model, feature_map, column_roles, num_records, MODELS_DIR, raw_file_path
        )
        
        # 3. Save synthetic CSV
        synth_path = os.path.join(SYNTHETIC_DIR, 'synthetic_data.csv')
        synth_df.to_csv(synth_path, index=False)
        
        TRAINING_STATE["percent"] = 94.0
        TRAINING_STATE["message"] = "Computing statistical benchmarks, Wasserstein distances, and privacy checks..."

        # 4. Comprehensive Evaluation
        real_df = pd.read_csv(raw_file_path) if raw_file_path.endswith('.csv') else pd.read_excel(raw_file_path)
        eval_report = evaluate_synthetic_data(real_df, synth_df, column_roles)
        
        # Save evaluation report and training history to JSON
        report_path = os.path.join(SYNTHETIC_DIR, 'evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(eval_report, f, indent=4)

        history_path = os.path.join(SYNTHETIC_DIR, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
            
        TRAINING_STATE["is_training"] = False
        TRAINING_STATE["percent"] = 100.0
        TRAINING_STATE["status"] = "complete"
        TRAINING_STATE["message"] = "Training & Synthesis complete!"

        return {
            "message": "Generation and evaluation complete", 
            "records_generated": num_records,
            "training_history": history,
            "quality_report": eval_report,
            "download_url": "/api/download",
            "report_download_url": "/api/download/report"
        }
        
    except Exception as e:
        TRAINING_STATE["is_training"] = False
        TRAINING_STATE["status"] = "error"
        TRAINING_STATE["message"] = f"Error: {str(e)}"
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

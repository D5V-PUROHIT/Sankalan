from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
import uvicorn
from analytics.profiling import profile_dataset
from analytics.preprocessing import clean_and_preprocess

app = FastAPI(title="Sankalan API", description="Deep Learning Synthetic Data API")

# Define absolute path to the frontend folder
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')

# API Endpoints
@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Endpoint to receive a dataset, save it to the raw folder, and initiate profiling.
    """
    if not file.filename.endswith(('.csv', '.xlsx')):
        return JSONResponse(status_code=400, content={"message": "Invalid file format. Only CSV and Excel supported."})
        
    # Ensure directory exists and save the uploaded file
    raw_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    file_path = os.path.join(raw_dir, file.filename)
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
        
    # Run data profiling on the saved file
    profile_result = profile_dataset(file_path)
    
    if "error" in profile_result:
        return JSONResponse(status_code=500, content={"message": f"Error profiling dataset: {profile_result['error']}"})
        
    # Run data cleaning and preprocessing for Deep Learning
    preprocess_result = clean_and_preprocess(
        file_path, 
        output_dir=os.path.join(os.path.dirname(__file__), 'data', 'processed'),
        models_dir=os.path.join(os.path.dirname(__file__), 'models', 'saved_models')
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
    Endpoint to retrieve data for EDA charts.
    Assumes a raw dataset has already been uploaded.
    """
    from analytics.eda import generate_eda_data
    import glob
    
    raw_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw')
    raw_files = glob.glob(os.path.join(raw_dir, '*'))
    if not raw_files:
        return JSONResponse(status_code=400, content={"message": "No raw dataset found. Please upload one first."})
        
    raw_file_path = raw_files[0]
    eda_data = generate_eda_data(raw_file_path)
    
    if eda_data.get("status") == "error":
        return JSONResponse(status_code=500, content={"message": f"EDA failed: {eda_data.get('message')}"})
        
    return eda_data

@app.post("/api/generate")
async def generate_synthetic_data(num_records: int = 5000, epochs: int = 50):
    """
    Endpoint to start the VAE training, generate synthetic data, and evaluate it.
    """
    from models.vae import train_vae
    from analytics.generation import generate_and_inverse_transform
    from analytics.evaluation import evaluate_synthetic_data
    import numpy as np
    import json
    import pandas as pd
    import glob
    
    # 1. Load data
    processed_dir = os.path.join(os.path.dirname(__file__), 'data', 'processed')
    models_dir = os.path.join(os.path.dirname(__file__), 'models', 'saved_models')
    synthetic_dir = os.path.join(os.path.dirname(__file__), 'data', 'synthetic')
    os.makedirs(synthetic_dir, exist_ok=True)
    
    try:
        X_train = np.load(os.path.join(processed_dir, 'X_train.npy'))
        with open(os.path.join(models_dir, 'column_roles.json'), 'r') as f:
            metadata = json.load(f)
            
        feature_map = metadata["feature_map"]
        column_roles = metadata["roles"]
        
        # Find raw file
        raw_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw')
        raw_files = glob.glob(os.path.join(raw_dir, '*'))
        if not raw_files:
            return JSONResponse(status_code=400, content={"message": "No raw dataset found."})
        raw_file_path = raw_files[0]
        
        # 2. Train VAE
        vae_model, history = train_vae(X_train, feature_map, epochs=epochs, batch_size=32)
        
        # 3. Generate & Inverse Transform
        synth_df = generate_and_inverse_transform(vae_model, feature_map, column_roles, num_records, models_dir, raw_file_path)
        
        # 4. Save synthetic data
        synth_path = os.path.join(synthetic_dir, 'synthetic_data.csv')
        synth_df.to_csv(synth_path, index=False)
        
        # 5. Evaluate
        real_df = pd.read_csv(raw_file_path) if raw_file_path.endswith('.csv') else pd.read_excel(raw_file_path)
        eval_report = evaluate_synthetic_data(real_df, synth_df, column_roles)
        
        return {
            "message": "Generation complete", 
            "records_generated": num_records,
            "quality_report": eval_report,
            "download_url": "/api/download"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"Generation failed: {str(e)}"})

from fastapi.responses import FileResponse

@app.get("/api/download")
async def download_synthetic():
    synth_path = os.path.join(os.path.dirname(__file__), 'data', 'synthetic', 'synthetic_data.csv')
    if os.path.exists(synth_path):
        return FileResponse(synth_path, media_type='text/csv', filename="synthetic_data.csv")
    return JSONResponse(status_code=404, content={"message": "File not found"})


# Mount the frontend static files at the root
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    print("Starting Sankalan Backend...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

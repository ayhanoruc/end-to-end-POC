# ENTRY POINT
"""serves 2 endpoint: 
    - training:
        - provide csv files for training
        - trains and saves model, saves training_history plot figure, and preprocessing object  
    - prediction:
        - provide csv file for prediction
        - returns predictions
"""

from fastapi import FastAPI, File, UploadFile
from src.training_pipeline import TrainingPipeline
from src.prediction_pipeline import PredictionPipeline
import csv 
import os 
import uvicorn
from fastapi.responses import JSONResponse

app = FastAPI()

async def read_file(file: UploadFile, training=False):
    # save the uploaded file as csv
    os.makedirs("uploads/prediction", exist_ok=True)
    os.makedirs("uploads/training", exist_ok=True)
    if not training:
        file_path = f"uploads/prediction/{file.filename}"
    else:
        file_path = f"uploads/training/{file.filename}"
    with open(file_path, "wb") as f:
        contents = await file.read()
        f.write(contents)
    print("training file uploaded succesfully")

    return file_path


@app.post("/training/")
async def training(file: UploadFile = File(...)):
    
    uploaded_file = await read_file(file, training=True)
    training_pipeline = TrainingPipeline()
    score = training_pipeline.run_pipeline(uploaded_file)
    response=  {"training_score": score}
    return JSONResponse(content=response)

@app.post("/prediction/")
async def prediction(file: UploadFile = File(...)):
    uploaded_file = await read_file(file, training=False)
    prediction_pipeline = PredictionPipeline()
    predictions = prediction_pipeline.run_pipeline(uploaded_file).tolist()
    response=  {"predictions": predictions}
    return JSONResponse(content=response)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
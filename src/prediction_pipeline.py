# prediction pipeline
import pandas as pd
import numpy as np 
import os
import pickle
from src.generic_funcs import load_yaml
from xgboost import XGBRegressor

class PredictionPipeline:
    def __init__(self):
        self.model_path = load_yaml("src/constants/pipeline.yaml")["Training_Pipeline"]["model_path"]
        self.model_name = load_yaml("src/constants/pipeline.yaml")["Training_Pipeline"]["model_name"]
        self.preprocessor_path = load_yaml("src/constants/pipeline.yaml")["Training_Pipeline"]["preprocessor_obj_path"]
        self.preprocessor_name = load_yaml("src/constants/pipeline.yaml")["Training_Pipeline"]["preprocessor_name"]
        self.model = self.load_model()
        self.preprocessor = self.load_preprocessor()

    def load_model(self):
        model = XGBRegressor()
        model.load_model(os.path.join(self.model_path, self.model_name))
        return model

    def load_preprocessor(self):
        with open(os.path.join(self.preprocessor_path, self.preprocessor_name), 'rb') as file:
            preprocessor = pickle.load(file)
        return preprocessor

    def preprocess_data(self, data):
        # we also need to check field names for compatibility
        data_np = np.array(data)  # Convert pandas DataFrame to NumPy array
        return self.preprocessor.transform(data_np)

    def predict(self, data):
        preprocessed_data = self.preprocess_data(data)
        predictions = self.model.predict(preprocessed_data)
        return predictions

    def run_pipeline(self, data_file_path):
        data = pd.read_csv(data_file_path)
        predictions = self.predict(data)
        return predictions

if __name__ == "__main__":
    prediction_pipeline = PredictionPipeline()
    predictions = prediction_pipeline.run_pipeline(r"C:\Users\ayhan\Desktop\cloud-POC\data\prediction_file\california_housing.csv")
    print(predictions)
    print(predictions.shape)

# training pipeline flow
"""
1- ingest dataset
2- preprocess: handle imbalance, impute missing values, scale features, etc.
3- initialize and fit model
4- evaluate model, save_metrics and plot training history
5- save model
"""

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
#import preprocessing pipeline from sklearn
from sklearn.pipeline import Pipeline
import pandas as pd 
import numpy as np 
from xgboost.callback import EarlyStopping


from src.generic_funcs import load_yaml

import os , sys 
import pickle


#print(load_yaml("src/constants/pipeline.yaml")["Training_Pipeline"]["model_name"])

class TrainingPipeline:
    def __init__(self):
        self.dataset = None
        self.x_train = None
        self.x_val = None
        self.y_train = None
        self.y_val = None
        
        self.model_hyperparameters = self.import_hyperparameters()
        self.model = self.initialize_model()
        self.evals_result = {}
        self.save_fig_path = load_yaml("src/constants/pipeline.yaml")["Training_Pipeline"]["training_history_figure_path"]
        self.model_path = load_yaml("src/constants/pipeline.yaml")["Training_Pipeline"]["model_path"]
        self.model_name = load_yaml("src/constants/pipeline.yaml")["Training_Pipeline"]["model_name"]
        

    def load_data(self,file_path:str)->None:
        
        self.dataset = pd.read_csv(file_path)
        return None


    def save_object(self, file_path, obj):
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)


    def preprocess(self)->None:
        #train-test split
        x_train, x_val, y_train, y_val = train_test_split(self.dataset.iloc[:,:-1], self.dataset.iloc[:,-1], test_size=0.2, random_state=42)
        
        #since we have all numeric, non-missing datapoints, we can skip imputation and one-hot encoding

        #perform scaling
        standard_scaler = StandardScaler()
        preprocessing_pipeline = Pipeline([
            ('standard_scaler', standard_scaler)
        ])


        self.x_train = preprocessing_pipeline.fit_transform(x_train)
        self.x_val = preprocessing_pipeline.transform(x_val)
        self.y_train = y_train
        self.y_val = y_val

        #save the preprocessor object
        path = load_yaml("src/constants/pipeline.yaml")["Training_Pipeline"]["preprocessor_obj_path"]
        obj_name = load_yaml("src/constants/pipeline.yaml")["Training_Pipeline"]["preprocessor_name"]
        os.makedirs(path, exist_ok=True)
        self.save_object(
            file_path=os.path.join(path, obj_name),
            obj=preprocessing_pipeline
        )
        print("Preprocessor object saved succesfully.")



    def initialize_model(self):
        model = XGBRegressor(
            n_estimators=self.model_hyperparameters["n_estimators"],
            max_depth=self.model_hyperparameters["max_depth"],
            learning_rate=self.model_hyperparameters["learning_rate"],
            eval_metric=self.model_hyperparameters["eval_metric"],
            callbacks=[EarlyStopping(rounds=10, save_best=True)],

        )
        return model

    def import_hyperparameters(self):
        return load_yaml("src/constants/pipeline.yaml")["Training_Pipeline"]["model_hyperparameters"]




    def train(self):
        evals_result = {}
        self.model.fit(
            self.x_train, self.y_train,
            eval_set=[(self.x_train, self.y_train), (self.x_val, self.y_val)],
            verbose=True
        )
        self.evals_result = self.model.evals_result()
        self.plot_training_history()

        os.makedirs(self.model_path, exist_ok=True)
        self.model.save_model(
            os.path.join(self.model_path, self.model_name),)
        

    def plot_training_history(self):
        train_rmse = self.evals_result['validation_0']['rmse']
        eval_rmse = self.evals_result['validation_1']['rmse']
        epochs = len(train_rmse)
        x_axis = range(0, epochs)
        
        plt.figure(figsize=(12, 8))
        plt.plot(x_axis, train_rmse, label='Train')
        plt.plot(x_axis, eval_rmse, label='Validation')
        plt.legend()
        
        plt.ylabel('RMSE')
        plt.xlabel('Epochs')
        plt.title('XGBoost RMSE')
        os.makedirs(self.save_fig_path, exist_ok=True)
        figure_file_name = os.path.join(self.save_fig_path, 'training_history.png')
        plt.savefig(figure_file_name)  # Save the figure
        plt.show() 

    

    def run_pipeline(self, data_file_path):
        self.load_data(data_file_path)
        self.preprocess()
        self.train()
        #return rmse score
        score = self.model.score(self.x_val, self.y_val)
        print("SUCCESFULL")
        return score

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline(load_yaml("src/constants/pipeline.yaml")["Data"]["data_file_path"])
    
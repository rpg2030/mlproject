import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname('E:/AI ML/Projects/ML project/src/components/data_ingestion.py'), "../../")))
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model,get_best_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("splitting training training and test data.")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "CatBoostRegressor":CatBoostRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "LinearRegression":LinearRegression(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "XGBRegressor":XGBRegressor(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
            }
            
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            
            model = get_best_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_report=model_report,models=models)
            
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=model)
            
            predicted = model.predict(X_test)
            r2 = r2_score(y_test,predicted)
            
            return (model,r2)
            
        except Exception as e:
            raise CustomException(e,sys)
            
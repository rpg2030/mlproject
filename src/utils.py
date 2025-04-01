import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    '''
    This functions will save pkl file.
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    '''
    This will read the pkl file.
    '''
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        logging.info("start the models training")
        report = {}
        for rec in range(len(list(models))):
            model = list(models.values())[rec]
            model.fit(X_train,y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)
            
            report[list(models.keys())[rec]] = test_model_score
            
        return report
        
    except Exception as e:
        raise CustomException(e,sys)
    
def hyperparameter_tuning(model_list,X_train,y_train,X_test,y_test):
    '''
    This will do Hyperparameter tuning.
    '''
    try:
        param_grid = {
            "CatBoostRegressor": {
                'iterations': [100, 200],
                'learning_rate': [0.01, 0.1],
                'depth': [4, 6, 8]
            },
            "AdaBoostRegressor": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0]
            },
            "GradientBoostingRegressor": {
                'n_estimators': [100, 200,300,400,500],
                'learning_rate': [0.01, 0.1,0.02,0.03,0.04],
                'max_depth': [3, 5, 7]
            },
            "RandomForestRegressor": {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            },
            "LinearRegression": {},  # No parameters for tuning
            "KNeighborsRegressor": {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            },
            "DecisionTreeRegressor": {
                'max_depth': [5, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            "XGBRegressor": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            },
            "Ridge": {
                'alpha': [0.1, 1.0,0.02,0.03,0.04]
            },
            "Lasso": {
                'alpha': [0.1, 1.0,0.02,0.03,0.04]
            }
        }

        hyper_model = {}
        for model in model_list:
            model_name = model.__class__.__name__
            if model_name in param_grid:
                params = param_grid[model_name]
                grid = GridSearchCV(estimator=model,param_grid=params,cv=5,n_jobs=-1,verbose=2,refit=True)
                grid.fit(X_train,y_train)
                
                y_train_grid = grid.predict(X_train)
                y_test_grid = grid.predict(X_test)
                
                train_grid_score = r2_score(y_train,y_train_grid)
                test_grid_score = r2_score(y_test,y_test_grid)
                
                hyper_model[model_name] = test_grid_score
        for model,score in hyper_model.items():
            if score == max(hyper_model.values()):
                best_hyper_model = model
        return (best_hyper_model,max(hyper_model.values()))      
            
    except Exception as e:
        raise CustomException(e,sys)
    
    
def get_best_model(model_report,models,X_train,y_train,X_test,y_test):
    '''
    This will give the best model after hyperparameter tuning and without hyperparameter tuning.
    '''
    logging.info("get the model which have highest score.")
    model_list = []
    
    # Get best model score
    best_model_score = max(model_report.values())
    for model_name,score in model_report.items():
        if score == best_model_score:
            model_before_hyper = model_name
    tolerance = 0.01
    
    # Get best model name
    best_model_names = [model_name for model_name,score in model_report.items() if abs(score-best_model_score) <= tolerance]
    
    for model in best_model_names:
        model_list.append(models[model])
        
    if best_model_score < 0.6:
        raise CustomException("No best model is found")
    else:
        logging.info(f"best model found....{model_list}")

    best_model={
        model_before_hyper:{
        "before_tuning":best_model_score
    }}
    
    logging.info("Hyperparameter tuning for best model.")
    hyper_model,hyper_score = hyperparameter_tuning(model_list=model_list,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)

    if hyper_model == model_before_hyper:
        best_model[hyper_model]["after_tuning"] = hyper_score
    else:
        best_model[hyper_model] = {"after_tuning":hyper_score}
    
    best_model_entry = max(best_model.items(), key=lambda x: max(x[1].values()))
    model = models[best_model_entry[0]]
    
    return model
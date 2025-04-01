import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('E:/AI ML/Projects/ML project/src/components/data_ingestion.py'), "../../")))
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer
@dataclass  
class DataIngestionConfig:
    '''
    whenever we do dataingestion there should be some inputs like 
    "where to save test data" , "Wehre to save test data" , "Where to save raw data"
    those kind of input create in this class.
    
    @dataclass  = it helps to define variable directly without using __init__().
    '''
    train_data_path: str=os.path.join("artifacts","train.csv")
    test_data_path: str=os.path.join("artifacts","test.csv")
    raw_data_path: str=os.path.join("artifacts","raw.csv ")
    '''
    this inputs that i giving to data ingestion components , so the data ingestion components
    knows where to save train , test and raw data.
    '''

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data Ingestion method.")
        
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Read the dataset as dataframe.")
            
            # make the artifacts folder
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            # make raw.csv
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info("Train Test Split Initiated")
            # make train dataset and test dataset
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            
            # make train.csv
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            
            # make test.csv
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("Ingestion of the data is completed.")

            return(self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)
            
        except Exception as e:
            raise CustomException(e,sys)
                
if __name__ == "__main__":
    ingestion_obj = DataIngestion()
    train_data,test_data = ingestion_obj.initiate_data_ingestion() 
    
    transformation_obj = DataTransformation()
    train_arr,test_arr,_ = transformation_obj.initiate_data__transformation(train_data,test_data)
    
    trainer_obj = ModelTrainer()
    print(trainer_obj.initiate_model_training(train_array=train_arr,test_array=test_arr))
                
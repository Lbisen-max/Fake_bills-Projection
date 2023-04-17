
import os
import sys
from src.exception import CustomException
# from src import logger
from logger import logging


import pandas as pd
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from sklearn.model_selection import train_test_split
from dataclasses import dataclass



@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifact','train.csv')
    test_data_path: str=os.path.join('artifact','test.csv')
    row_data_path: str=os.path.join('artifact','data.csv')



class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()


    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv(r'D:\ML Projects\fakebills\Notebook\data\Fake Bills.csv',sep = ";")

            logging.info("Read the datasent as dataframe")

            os.mkdir(os.path.dirname(self.ingestion_config.train_data_path))

            df.to_csv(self.ingestion_config.row_data_path,index=False,header=True)

            logging.info("train test split initiated")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=41)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data compeleted")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e,sys)
        



if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()


    data_transformation=DataTransformation()
    data_transformation.initiate_data_trasnformation(train_data,test_data)





  






    

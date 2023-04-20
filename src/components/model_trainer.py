import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import os
import sys

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_score, recall_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dataclasses import dataclass



from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from logger import logging
from src.utils import save_objec,evaluate_models
import os

@dataclass
class ModelTrainerConfig():
    trained_model_file_path= os.path.join("artifact","model.pkl")



class Model_Trainer:
    def __init__ (self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_arry,test_array):
        try:
            logging.info("Spliting the train and test input data")

            X_train,y_train,X_test,y_test = (train_arry[:,:-1],
                                          train_arry[:,-1],
                                          test_array[:,:-1],
                                          test_array[:,-1])
            
            models = { "logistic regressionn": LogisticRegression(),
                      "KNN classifier" : KNeighborsClassifier(),
                      "Decesion tree classifier" : DecisionTreeClassifier(),
                      "Random classifier" : RandomForestClassifier(),
                      "Ada boost classifier" : AdaBoostClassifier(),
                      "gradient boosting classifier" : GradientBoostingClassifier(),}
          
               
            model_report : dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                                   models=models
                                                   )
            logging.info("Model trained and evaluated")

            best_model_score = max(sorted(model_report.values()))

            # to get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score<80:
                print("no best model")
                
            logging.info(f"Found model on traning dataset")

            

            save_objec(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model

            )
            
        except Exception as e:
            raise CustomException(e,sys)









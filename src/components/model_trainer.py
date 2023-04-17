# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# %matplotlib inline
# import warnings
# warnings.filterwarnings('ignore')

# from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_score, recall_score
# from sklearn import metrics
# from sklearn.linear_model import LogisticRegression
# from sklearn.cluster import KMeans
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import VotingClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score


# from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline

# from src.exception import CustomException
# from logger import logging
# from src.utils import save_objec
# import os

# @dataclass
# class ModelTrainerConfig():
#     trained_model_file_path= os.path.join("artifact","model.pkl")



# class Model_Trainer:
#     def __init__ (self):
#         self.model_trainer_config=ModelTrainerConfig()


#     def initiate_model_trainer(self,train_arry,test_array,preprocessor_path):
#         try:
#             logging.info("Spliting the train and test input data")

#             X_train,y_train,X_test,y_test(train_arry[:,:1],
#                                           train_arry[:,1],
#                                           test_array[:,:1],
#                                           test_array[:,1])
            
#             models = {
#                 ""
#             }
#         except:
#             pass









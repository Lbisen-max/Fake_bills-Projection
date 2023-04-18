import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder




from src.exception import CustomException
from logger import logging
import os

from src.utils import save_objec

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path=os.path.join('artifact',"preprocessor.pkl")



class DataTransformation:
    def __init__(self):
        self.Data_transformation_config=DataTransformationConfig()


    def grt_data_transformer_object(self):

        '''
        This function is responsible for data transformation

        '''

        try:
            num_col = ['diagonal','height_left','height_right','margin_low','margin_up','length']
            cat_col = ['is_genuine']
            
            pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="mean")),
                ("scaler",StandardScaler())
                ]
            )

            pipeline_label = Pipeline(
                steps=[
                ("label",LabelEncoder())
                ]
            )


            logging.info("Pipeline created and completed")

            preprocessor = ColumnTransformer(
                [
                ('pipeline',pipeline,num_col),
                ('pipeline_label',pipeline_label,cat_col),
                ],remainder='passthrough'
            )

            # preprocessor_label = ColumnTransformer([
            #     ('pipeline_label',pipeline_label,cat_col)
            # ])

            return preprocessor
        

        except Exception as e:

            raise CustomException(e,sys)
    

               

    def initiate_data_trasnformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtainng preprocessor objct")

            preprocessing_obj=self.grt_data_transformer_object()


            target_col_name = "is_genuine"
            num_value = ["diagonal","height_left","height_right","margin_low","margin_up","length"]


            input_feature_train_df = train_df.drop(columns=[target_col_name],axis=1)
            target_feature_train_df = train_df[target_col_name]



            input_feature_test_df = test_df.drop(columns=[target_col_name],axis=1)
            target_feature_test_df = test_df[target_col_name]


            logging.info(f"Applying preprocessing object on traing dataframe and testing dataframe.")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.fit_transform(input_feature_test_df)

            input_target_feature_train_arr=preprocessing_obj.fit_transform(target_feature_train_df)
            input_target_feature_test_arr=preprocessing_obj.fit_transform(target_feature_test_df)



            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_objec(

                file_path=self.Data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,test_arr,self.Data_transformation_config.preprocessor_ob_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)



            


        


            
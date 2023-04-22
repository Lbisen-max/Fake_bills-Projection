import sys
import pandas as pd
from src.exception import CustomException
from logger import logging
from src.utils import save_objec,evaluate_models,load_object

class PredictPipeline:
    def __init__(self):
        pass


    def predict(self,features):

        try:
            model_path = 'artifact\model.pkl'
            preprocessor_path = 'artifact\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scale = preprocessor.transform(features)
            preds = model.predict(data_scale)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)




class cus:
    def __init__(self):
        pass


class CustomData:
    def __init__(self ,
                 diagonal : int,
                 height_left : int,
                 height_right : int,
                 margin_low : int ,
                 margin_up : int ,
                 length : int):
        self.diagonal = diagonal
        self.height_left = height_left
        self.height_right = height_right
        self.margin_low = margin_low
        self.margin_up = margin_up
        self.length = length

    def get_data_as_frame(self):
        try:
            custom_data_input_dict = {
                "diagonal" : [self.diagonal],
                "height_left" : [self.height_left],
                "height_right" : [self.height_right],
                "margin_low" : [self.margin_low],
                "margin_up" : [self.margin_up],
                "length" : [self.length]

            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)
    





        


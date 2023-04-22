from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from logger import logging
from src.utils import save_objec,evaluate_models
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
import os

application=Flask(__name__)

app=application

## route for homepage 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            diagonal = float(request.form.get('diagonal')),
            height_left = float(request.form.get('height_left')),
            height_right = float(request.form.get('height_right')),
            margin_low = float(request.form.get('margin_low')),
            margin_up = float(request.form.get('margin_up')),
            length = float(request.form.get('length')),
        )

        pred_df = data.get_data_as_frame()
        print(pred_df)


        predict_pipelie = PredictPipeline()
        results=predict_pipelie.predict(pred_df)
        return render_template('home.html',results=results[0])
    



if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
    

        


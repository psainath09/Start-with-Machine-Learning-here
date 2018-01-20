# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 12:20:58 2018

@author: satya naidu
"""
import numpy as np
from flask import Flask
from sklearn.externals import joblib
import pandas as pd
classi=joblib.load("./logistic_regression_model.pkl")

t=pd.read_csv("test.csv")

app=Flask(__name__)


@app.route('/predict', methods=['POST'])

def predict():
    if request.method=='POST':
        try:
            data=request.get_json()
            
        except ValueError:
            return jsonify("please enter number")
        return jsonify(classi.predict(t[3,4]).tolist())
app.add_url_rule("/","hello",hello_world)

if __name__=='__main__':
    app.run()
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import pickle
import joblib
from flask import Flask, request, render_template,jsonify
import pybase64
app = Flask(__name__)
# pickle_in = open(r"XGB_Model.sav","rb")

# filename = 'XGB_Model.sav'
# with open(filename, 'rb') as f:
#     model = joblib.load(f)

# model = pickle.load(open(r"XGB_Model.sav","rb"))
model_xgb = pickle.load(open(r"XGB_Model.sav","rb"))
makel_label = pd.read_csv(r"Make.csv")
model_label = pd.read_csv(r"Model.csv") 
city_label = pd.read_csv(r"City.csv")

def price_pred(mil,make,model,city,model_xgb):
    
    makel_label['Make'] = [str(i).lower().strip() for i in makel_label['Make']]
    model_label['Model'] = [str(i).lower().strip() for i in model_label['Model']]
    city_label['City'] = [str(i).lower().strip() for i in city_label['City']]

    reqcols = ['Mileage', 'Make', 'Model', 'City']



    if make in list(makel_label['Make'].unique()):
        makevalue = makel_label[makel_label['Make']==str(make).lower().strip()]['Label'][0]
    else:
        makevalue = 999
        
    if model in list(model_label['Model'].unique()):
        modelvalue = model_label[model_label['Model']==str(model).lower().strip()]['Label'][0]
    else:
        modelvalue = 999
        
    if city in list(city_label['City'].unique()):
        cityvalue = city_label[city_label['City']==str(city).lower().strip()]['Label'][0]
    else:
        cityvalue = 999
        
    testx = pd.DataFrame([[mil,makevalue,modelvalue,cityvalue]])
    testx.columns = ['Mileage', 'Make', 'Model', 'City']    
    prediction = model_xgb.predict(testx)

    result_price = prediction[0]

    return result_price


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("home.html")


@app.route('/input', methods=['GET', 'POST'])
def input():
    if request.method=='POST':
        mil = request.form.get('Mileage')#int
        make = request.form.get('Make')#char
        model = request.form.get('Model')#char
        city = request.form.get('City')#char
        price_value=price_pred(mil,make,model,city,model_xgb)

        return render_template("home.html",price=price_value)


if __name__ == '__main__':
    app.run(debug=True) 
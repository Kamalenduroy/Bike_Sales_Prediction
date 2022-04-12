# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 04:23:13 2022

@author: rkama
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

def predicting_sale(feature_dict):

    #preprocessing the input data

    print(feature_dict.keys())
    
    yr = int(feature_dict['yr'])
    
    if yr == 2018:
        yr = 0
    else:
        yr = 1
        
    workingday = int(feature_dict['workingday'])
    
    temp = int(feature_dict['temp'])
    hum = int(feature_dict['hum'])
    windspeed = int(feature_dict['windspeed'])

    if feature_dict['season'] == 'Spring':
        season_spring = 1
        season_winter = 0
    elif feature_dict['season'] == 'winter':
        season_spring = 0
        season_winter = 1
    else:
        season_spring = 0
        season_winter = 0

    month_july = 0
    month_september = 0

    if feature_dict['month'] == 'Jul':
        month_july = 1
    elif feature_dict['month'] == 'Sep':
        month_september = 1

    if feature_dict['day'] == 'Sat':
        weekday_saturday = 1
    else: 
        weekday_saturday = 0

    weathersit_Cloudy = 0
    weathersit_Light_Rain = 0

    if feature_dict['weathercond'] == 'Cloudy':
        weathersit_Cloudy = 1
    elif feature_dict['weathercond'] == 'Light Rain':
        weathersit_Light_Rain = 1

    scaled_output = feature_scaling.transform(np.array([temp,hum,windspeed]).reshape(1,-1))
    temp, hum, windspeed = scaled_output[0,0], scaled_output[0,1], scaled_output[0,2]
    
    print("debug msg 1:")
    print("The feature values are: ")
    print(np.array([1,yr, workingday, temp, hum, windspeed, season_spring, 
                                          season_winter, month_july, month_september, weekday_saturday, 
                                          weathersit_Cloudy, weathersit_Light_Rain]).reshape(1,-1))
    
    
    sales_amount_scaled = model.predict(np.array([1,yr, workingday, temp, hum, windspeed, season_spring, 
                                          season_winter, month_july, month_september, weekday_saturday, 
                                          weathersit_Cloudy, weathersit_Light_Rain]).reshape(1,-1))
    
    print("Debug msg 2")
    print("The sale amount before reverse transform is ", sales_amount_scaled)
    

    sales_amount = int(target_scaling.inverse_transform(sales_amount_scaled.reshape(1,1)))

    return sales_amount

app = Flask(__name__)

model = pickle.load(open('bike_model.pkl','rb'))
feature_scaling = pickle.load(open('scaling.pkl','rb'))
target_scaling = pickle.load(open('scaling_target.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')
                           
    
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    feature_values = [x for x in request.form.values()]
    features = ['yr','workingday','temp','hum','windspeed','season','weathercond','month','day']
    feature_dict = {k:v for (k,v) in zip(features,feature_values)}
    print("feature dictionary input is: ",feature_dict)
    sales_amount = predicting_sale(feature_dict)

    return render_template('index.html', prediction_text='Expected Sales Amount = {}'.format(sales_amount))


if __name__ == "__main__":
    app.run(debug=True)                           




























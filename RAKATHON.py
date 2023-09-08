# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img
from keras.models import load_model
import os
from flask import Flask, flash, request, redirect, url_for,jsonify



from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from flask import Flask, render_template, request

import tempfile
app=Flask(__name__)

result  = {0:"Tier_damage",1:"Tier_healthy",2:"Headlight_Damage",3:"Headlight_healthy",4:"Glass_Damage",5:"Glass_healthy",6:"Dent_damage",7:"Dent_Healthy"}
MODEL_PATH =   'C:/Users/NAEEM/RAKATHON.h5'

model_dl = load_model(MODEL_PATH)
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img_file = request.files['image']
        
        # Save the temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            img_file.save(temp.name)
            temp_path = temp.name
        
        # Preprocess the image for the model
        img = image.load_img(temp_path, target_size=(224, 224))
        img_arr = image.img_to_array(img)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = np.vstack([img_arr])
        # Make a prediction with the model
        Result = model_dl.predict(img_arr)
        prediction_index = np.argmax(Result)
        prediction = result[prediction_index]
        
        # Remove the temporary file
        os.remove(temp_path)
        
        # Return the prediction as a JSON response
        response = {'prediction': str(prediction)}
        return jsonify(response)
    else:
        return "No image passed UBED"


if __name__ == '__main__':
    app.run(  port =5000, host='10.196.220.46')

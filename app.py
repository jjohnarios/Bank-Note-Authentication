# -*- coding: utf-8 -*-
"""

@author: jjohnarios
"""

from flask import Flask, request
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.keras.backend import set_session
import flasgger
from flasgger import Swagger

# Ignore Tensorflow warnings
tf.get_logger().setLevel('INFO')

app=Flask(__name__)
Swagger(app) # Generate UI using flasgger

# Needed for keras
sess = tf.Session()
graph = tf.get_default_graph()

# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras! 
# Otherwise, their weights will be unavailable in the threads after the session there has been set
set_session(sess)

model=load_model("model.h5")



# route page
@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict')
def predict_bank_note_authentication():
    
    """Bank Note Authentication.
    This is using docstrings for specifications.
    ---
    parameters:
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
            
    """
    global graph
    global sess
    # Get all feature variables
    variance=request.args.get("variance")
    skewness=request.args.get("skewness")
    curtosis=request.args.get("curtosis")
    entropy=request.args.get("entropy")
    
    # make its shape (4,)
    note=np.array([variance,skewness,curtosis,entropy]).reshape((1,-1))
    
    with graph.as_default():
        set_session(sess)
        prediction=model.predict(note)
        prediction=round(float(prediction)) # Round to closest integer 0 or 1
    
    # Prediction value into either 'Fake' or 'Authentic'
    result=""
    if prediction==0:
        result="Fake"
    elif prediction==1:
        result="Authentic"
    
    return "Bank Note is: {}".format(result)

# Post to send formData to server
@app.route('/predict_file',methods=["POST"])
def predict_bank_note_file():
    '''Bank Note Authentication.
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        
    responses:
        200:
            description: The output values
    '''
    global graph
    global sess

    df_test=pd.read_csv(request.files.get("file"))
    
    with graph.as_default():
        set_session(sess)
        prediction=model.predict(df_test)
    
    results=[]
    # Every prediction value into either 'Fake' or 'Authentic'
    for p in list(prediction):
        pred=round(float(p))
        if pred==0:
            results.append("Fake")
        elif pred==1:
            results.append("Authentic")
    
    
    return "Bank Notes are: {}".format(results)


if __name__=='__main__':
    app.run(host='0.0.0.0')

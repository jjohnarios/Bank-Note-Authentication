# -*- coding: utf-8 -*-
"""

@author: jjohnarios
"""

from flask import Flask, request
import pandas as pd
import numpy as np
from keras.models import load_model
import tensorflow as tf
tf.get_logger().setLevel('INFO')
from tensorflow.python.keras.backend import set_session

app=Flask(__name__)

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
    return "Welcome ALL"

@app.route('/predict')
def predict_bank_note_authentication():
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
    
    # Ισως θελει να περαστεί από συνάρτηση για να δίνει ακριβώς 0 ή 1.
    
    return "Prediction: {}".format(prediction)

@app.route('/predict_file',methods=["POST"])
def predict_bank_note_file():
    global graph
    global sess

    df_test=pd.read_csv(request.files.get("file"))
    
    with graph.as_default():
        set_session(sess)
        prediction=model.predict(df_test)    
    
    # Ισως θελει να περαστεί από συνάρτηση για να δίνει ακριβώς 0 ή 1.
    
    return "Predicted values for csv are: {}".format(list(prediction))


if __name__=='__main__':
    app.run()

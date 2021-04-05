# -*- coding: utf-8 -*-
"""
This is the same app implementation but with Streamlit. Everything Flask-or-Flasgger related is missing

@author: jjohnarios
"""


import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.keras.backend import set_session
import streamlit as st

# Ignore Tensorflow warnings
tf.get_logger().setLevel('INFO')


# Needed for keras
sess = tf.Session()
graph = tf.get_default_graph()

# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras! 
# Otherwise, their weights will be unavailable in the threads after the session there has been set
set_session(sess)

model=load_model("model.h5")




def predict_bank_note_authentication(variance,skewness,curtosis,entropy):
    
    """Bank Note Authentication.
    Predicts if the bank note is Authentic or Fake based on its 4 features given.
            
    """
    global graph
    global sess
    
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


def st_main():
    '''
    Setting up Streamlit.
    '''
    st.title("Bank Note Authentication: A simple Neural Network approach.")
    st.markdown("Welcome to my web application. For this project, I used the Bank Note Authentication [dataset](https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data) from Kaggle.")
    
    #Add Img
    st.image("https://images.unsplash.com/photo-1600007283728-22abc97b9318?ixid=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&ixlib=rb-1.2.1&auto=format&fit=crop&w=1050&q=80")
    
    st.header("Give Features and Predict")
    # Reading the 4 features
    variance=st.number_input("Variance",help='Give a float or integer.')
    skewness=st.number_input("Skewness",help='Give a float or integer.')
    curtosis=st.number_input("Curtosis",help='Give a float or integer.')
    entropy=st.number_input("Entropy",help='Give a float or integer.')
    
    # Customized Button
    html='''<style>
    
    .stButton>button {
          border: 2px solid black;
          background-color: #8FBC8F;
          color: black;
          padding: 14px 28px;
          font-size: 20px;
          font-weight: bold;
          border-radius: 0 10px
    }
    </style>
    
    '''
    st.markdown(html, unsafe_allow_html=True)
    
    
    
    prediction=""
    emoji=":+1:"
    if st.button("Predict"):
        prediction=predict_bank_note_authentication(variance,skewness,curtosis,entropy)
        if "Fake" in prediction:
            emoji=":-1:"
        st.success("Prediction -  {0} {1}".format(prediction,emoji))


if __name__=='__main__':
    st_main()
    

        

    
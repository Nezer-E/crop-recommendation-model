import pandas as pd 
import numpy as np
#import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import streamlit as st
#import seaborn as sns

#url_content = "https://raw.githubusercontent.com/Nezer-E/machine-learning-datasets/main/croprecommendation.csv"

def inputprocessor():
    #Input parameters
    st.sidebar.header("CROP RECOMENDATION MODEL INPUT PARAMETERS\n")
    Nitrogen = st.sidebar.text_input("Amount of Nitrogen (ppm)", 90)
    Phosphorus = st.sidebar.text_input("Amount of Phosphorus (ppm)", 60)
    Potassium = st.sidebar.text_input("Amount of Potassium(ppm)", 60)
    Temperature = st.sidebar.text_input("Temperature (celcius)", 26)
    Humidity = st.sidebar.text_input("Humidity (%)", 50)
    pH = st.sidebar.slider("pH", 0, 14, 7)
    Rainfall = st.sidebar.text_input("Monthly Rainfall(mm)", 200)

    test_data = pd.DataFrame({
    'N': [Nitrogen],
    'P': [Phosphorus],
    'K': [Potassium],
    'temperature': [Temperature],
    'humidity': [Humidity],
    'ph': [pH],
    'rainfall': [Rainfall],
})
 
    st.header("CROP RECOMMENDATION MODEL")
    st.subheader("User Input Data")
    st.write(test_data)
    st.subheader("Results will appear in this section")

    train_data = pd.read_csv("modifiedcroptrain.csv")
        
   

    X_train = train_data.drop('label', axis =1)
    Y_train = pd.DataFrame({
        'label': train_data['label']
    })
    button = st.sidebar.button("Predict")
    if button:
        #conversion to numpy array
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        test_data = np.array(test_data)
        
        #feature scaling
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        test_data = scaler.transform(test_data)
        
        #accuracy score = 99.3%
        classifier = XGBClassifier(learning_rate = 0.25, max_depth = 7, n_estimators = 60, subsample = 0.9, colsample_bytree = 0.3)
        
        
        classifier.fit(X_train, Y_train)

        prediction = classifier.predict(test_data)

        crops = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee' ]
        
        
        for i in range(22):
            if prediction == [i]:
                prediction = crops[i]
                st.write(f"The recommended crop is {prediction}")

inputprocessor()

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

    train_data = pd.read_csv("Croprecommendation.csv")
        
    train_data = train_data.replace('rice', 0)
    train_data = train_data.replace('maize', 1)
    train_data = train_data.replace('chickpea', 2)
    train_data = train_data.replace('kidneybeans', 3)
    train_data = train_data.replace('pigeonpeas', 4)
    train_data = train_data.replace('mothbeans', 5)
    train_data = train_data.replace('mungbean', 6)
    train_data = train_data.replace('blackgram', 7)
    train_data = train_data.replace('lentil', 8)
    train_data = train_data.replace('pomegranate', 9)
    train_data = train_data.replace('banana', 10)
    train_data = train_data.replace('mango', 11)
    train_data = train_data.replace('grapes', 12)
    train_data = train_data.replace('watermelon', 13)
    train_data = train_data.replace('muskmelon', 14)
    train_data = train_data.replace('apple', 15)
    train_data = train_data.replace('orange', 16)
    train_data = train_data.replace('papaya', 17)
    train_data = train_data.replace('coconut', 18)
    train_data = train_data.replace('cotton', 19)
    train_data = train_data.replace('jute', 20)
    train_data = train_data.replace('coffee', 21)

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

        params = {
            'learning_rate':[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
            'max_depth': [1,2,3,4,5,6,7,8.9,10],
            'n_estimators': [10,20,30,40,50,60,70,80,90,100],
            'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }

        #accuracy score = 99.3%
        classifier = XGBClassifier(learning_rate = 0.25, max_depth = 7, n_estimators = 60, subsample = 0.9, colsample_bytree = 0.3, )
        
        
        classifier.fit(X_train, Y_train)

        prediction = classifier.predict(test_data)

        crops = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee' ]
        
        
        for i in range(22):
            if prediction == [i]:
                prediction = crops[i]
                st.write(f"The recommended crop is {prediction}")

inputprocessor()
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

 ##Load the ANN train model, scakler andd one hot model
model = tf.keras.models.load_model('model.h5')

##load the encoder and scaler
with open('labe_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
with open('onehot_encoder_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

##streamlit app
st.title('Customer Churn Prediction')

##User input
geography = st.selectbox('Geography',label_encoder_geo.categories_[0])
gender = st.selectbox("Gender",label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimited_salary = st.number_input('Estimited Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number Of Products', 1, 4)
has_cr_card = st.selectbox("Has Credit Card",[0,1])
is_active_member = st.selectbox("Is Active Member",[0,1]) 

input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance':[balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member], 
    'EstimatedSalary': [estimited_salary]
})

geo_encoded= label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=label_encoder_geo.get_feature_names_out(['Geography']))

input_df = pd.concat([input_data.reset_index(drop=True),geo_encoded_df], axis =1)


##Sacle the data
input_data_sacled = scaler.transform(input_df)

##Prediction

prediction = model.predict(input_data_sacled)
prediction_proba = prediction[0][0]
prediction_probability = prediction_proba*100

st.write(f"Churn Probability: {prediction_probability:,.0f}% ")

if prediction_proba > 0.5:
    st.write("The Customer will probably Churn")
else:
    st.write("The Customer will not likely to Churn")
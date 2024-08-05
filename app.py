#!/usr/bin/env python
# coding: utf-8

# In[23]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('dataset.csv')

df = df.drop(['Loan_ID'], axis=1)



## Fill the null values of numerical datatype
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())

## Fill the null values of object datatype
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])

df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])


# Preprocess the data
# Perform necessary data cleaning, feature engineering, and encoding steps
categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

# Ordinal Encoding for categorical features
ordinal_encoder = OrdinalEncoder()
df[categorical_features] = ordinal_encoder.fit_transform(df[categorical_features])

# Scaling for numerical features
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Split the data into features and target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Streamlit app
def main():
    st.title('Loan Approval Prediction')
    
    # Input fields for user to enter loan application details
    gender = st.selectbox('Gender', ['Male', 'Female'])
    married = st.selectbox('Married', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
    education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
    income = st.number_input('Applicant Income', min_value=0)
    co_income = st.number_input('Coapplicant Income', min_value=0)
    loan_amount = st.number_input('Loan Amount (in thousands)', min_value=0)
    loan_term = st.number_input('Loan Amount Term (in months)', min_value=0)
    credit_history = st.selectbox('Credit History', [0, 1])
    property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])
    
    # Create a dictionary to store the user inputs
    user_input = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': income,
        'CoapplicantIncome': co_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }
    
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Preprocess user input
    input_df[categorical_features] = ordinal_encoder.transform(input_df[categorical_features])
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])
    
    # Create a button to trigger the prediction
    if st.button('Predict'):
        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions on user input
        prediction_proba = model.predict_proba(input_df)
    
        # Print the predicted probabilities for debugging
        st.write("Predicted Probabilities:")
        st.write(prediction_proba)
    
        # Get the predicted class based on the highest probability
        predicted_class = np.argmax(prediction_proba)
    
        # Display the prediction result
        if predicted_class == 1:
            st.success('Loan Approved!')
        else:
            st.error('Loan Denied.')

if __name__ == '__main__':
    main()


# In[ ]:





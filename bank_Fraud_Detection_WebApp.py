import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

# Load trained Random Forest model and LabelEncoders from .pkl files
@st.cache(allow_output_mutation=True)
def load_model_and_encoders():
    rf_model = joblib.load("rf_smote_model.pkl")  # Your trained RandomForest model file
    le_dict = joblib.load("label_encoders.pkl")   # Dict of LabelEncoders for categorical features
    return rf_model, le_dict

rf_model, le_dict = load_model_and_encoders()


st.title("Bank Fraud Detection Web App")
st.markdown("""
Enter transaction details below to predict whether it is "Fraud" or "Non-Fraud".
""")

# User Input Form
with st.form(key='transaction_form'):
    TransactionAmount = st.number_input("Transaction Amount", min_value=0.0, step=0.01)
    TransactionDate = st.date_input("Transaction Date")
    Location = st.text_input("Location")
    IP_Address = st.text_input("IP Address")
    MerchantID = st.text_input("Merchant ID")
    PreviousTransactionDate = st.date_input("Previous Transaction Date")

    submit_button = st.form_submit_button(label='Predict')

def preprocess_input(new_transaction, le_dict):
    # Convert datetime columns to datetime dtype
    new_transaction['TransactionDate'] = pd.to_datetime(new_transaction['TransactionDate'])
    new_transaction['PreviousTransactionDate'] = pd.to_datetime(new_transaction['PreviousTransactionDate'])

    # Label encode categorical columns using pre-fitted encoders
    for col, le in le_dict.items():
        if col in new_transaction.columns:
            # Map unknown categories to first known category to prevent errors
            new_transaction[col] = new_transaction[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            new_transaction[col] = le.transform(new_transaction[col])
    # Select features used in model training (adjust columns as per your training data)
    selected_features = ['TransactionAmount', 'TransactionDate', 'Location', 'IP Address', 'MerchantID', 'PreviousTransactionDate']

    preprocessed = new_transaction[selected_features].copy()
    return preprocessed

if submit_button:
    input_dict = {
        "TransactionAmount": [TransactionAmount],
        "TransactionDate": [TransactionDate],
        "Location": [Location],
        "IP Address": [IP_Address],
        "MerchantID": [MerchantID],
        "PreviousTransactionDate": [PreviousTransactionDate]
    }
    new_trans_df = pd.DataFrame(input_dict)
    processed_df = preprocess_input(new_trans_df, le_dict)

    # Predict and display result
    prediction = rf_model.predict(processed_df)[0]
    if prediction == 1:
        st.error("🚨 Fraud Transaction Detected!")
    else:
        st.success("✅ Transaction is Non-Fraud.")


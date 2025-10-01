import streamlit as st
import numpy as np
import pandas as pd

import joblib


kmeans = joblib.load("k_means_0.1.0.pkl")

scaler = joblib.load("scaler_1.0.0.pkl")

st.title("Customer Segmentation App")
st.write("Enter customer details to predict segment ")

age = st.number_input("Age", min_value = 18,  max_value = 100, value = 35)

income = st.number_input("Income", min_value = 0,  max_value = 200_000, value = 50_000)

recency = st.number_input("Recency (days since last purchase)", min_value = 0,  max_value = 1000, value = 35)  

total_spending = st.number_input("Total Spending (sum of purchases)", min_value = 0,  max_value = 5000, value = 1000)

num_web_purchases = st.number_input("Number of Web Purchases", min_value = 0,  max_value = 100, value = 10)

num_store_purchases = st.number_input("Number of Store Purchases", min_value = 0,  max_value = 50, value = 10)

num_web_visit = st.number_input("Number of Web Visit", min_value = 0,  max_value = 50, value = 1)


input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Recency": [recency],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "Total_Spending": [total_spending],
    "NumWebVisitsMonth":[num_web_visit]
})

features = ["Age", "Income",'Recency',"NumWebPurchases","NumStorePurchases",'Total_Spending','NumWebVisitsMonth']
input_data = input_data[features]
input_scaler = scaler.transform(input_data)

if st.button("Predict Segment"):
    cluster = kmeans.predict(input_data)[0]

    st.success(f"Predicted Segment: Cluster {cluster}")

    # st.write("""
    #     Cluster 0 : High budget, web visitors   
    #     Cluster 1 : High spending 
    #     Cluster 2 : Web visitors

    #      """)

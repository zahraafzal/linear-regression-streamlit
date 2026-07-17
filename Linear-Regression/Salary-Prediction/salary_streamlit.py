import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import os

st.title("Salary Prediction")

# Load and prepare data (cached)
@st.cache_resource
def load_and_train_model():
    # Try different file paths
    possible_paths = [
        'Salary_Data.csv',
        Path(__file__).parent / 'Salary_Data.csv',
        'salary_data.csv',
        Path(__file__).parent / 'salary_data.csv'
    ]
    
    df = None
    for path in possible_paths:
        try:
            df = pd.read_csv(path)
            break
        except:
            continue
    
    if df is None:
        st.error("CSV file not found. Please upload Salary_Data.csv to GitHub!")
        st.stop()
    
    # Drop missing values
    df = df.dropna()
    
    # Create and fit Label Encoders
    le_gender = LabelEncoder()
    le_education = LabelEncoder()
    le_job = LabelEncoder()
    
    df["Gender"] = le_gender.fit_transform(df["Gender"])
    df["Education Level"] = le_education.fit_transform(df["Education Level"])
    df["Job Title"] = le_job.fit_transform(df["Job Title"])
    
    # Train Model
    X = df.drop("Salary", axis=1)
    y = df["Salary"]
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Return model and encoders
    encoders = {
        'gender': le_gender,
        'education': le_education,
        'job': le_job
    }
    
    return model, encoders

# Load model and encoders
model, encoders = load_and_train_model()

# Get unique values from encoders for dropdowns
gender_options = list(encoders['gender'].classes_)
education_options = list(encoders['education'].classes_)
job_options = list(encoders['job'].classes_)

# Predict salary
st.divider()
st.markdown("#### Enter the Details Below")

age = st.number_input("Enter Age:", min_value=18, max_value=70, value=30)
gender = st.selectbox("Select Gender:", gender_options)
education = st.selectbox("Select Education Level:", education_options)
job_title = st.selectbox("Select Job Title:", job_options)
experience = st.number_input("Years of Experience:", min_value=0, max_value=50, value=5)

if st.button("Predict Salary"):
    sample_data = {
        "Age": age,
        "Gender": gender,
        "Education Level": education,
        "Job Title": job_title,
        "Years of Experience": experience
    }
    
    sample_df = pd.DataFrame([sample_data])
    
    # Transform using cached Label Encoders
    sample_df["Gender"] = encoders['gender'].transform(sample_df["Gender"])
    sample_df["Education Level"] = encoders['education'].transform(sample_df["Education Level"])
    sample_df["Job Title"] = encoders['job'].transform(sample_df["Job Title"])
    
    # Predict
    prediction = model.predict(sample_df)
    
    st.success(f"Predicted Salary: ${prediction[0]:,.2f}")

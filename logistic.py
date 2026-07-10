import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score


st.title("Diabetes Prediction")
df = pd.read_csv("diabetes.csv")

X = df.drop(["Outcome"], axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)
model = LogisticRegression(C=0.5, max_iter=2000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
st.subheader("Enter Patient Information")

pregnancies = st.number_input("Pregnancies", 0, 20, 0)
glucose = st.number_input("Glucose", 0, 300, 100)
blood_pressure = st.number_input("Blood Pressure", 0, 200, 80)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 35)
insulin = st.number_input("Insulin", 0, 900, 0)
bmi = st.number_input("BMI", 0.0, 70.0, 25.6)
diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    0.0,
    3.0,
    0.201
)
age = st.number_input("Age", 1, 120, 30)
if st.button("Predict"):

    data = [[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]]

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Prediction: Diabetes")
    else:
        st.success("Prediction: No Diabetes")



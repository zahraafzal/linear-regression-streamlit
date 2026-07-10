import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

st.title("House Price Prediction")


@st.cache_resource
def load_and_train_model():
    df = pd.read_csv("Housing.csv")
    
    # Label Encoders banao aur fit karo
    le_mainroad = LabelEncoder()
    le_guestroom = LabelEncoder()
    le_basement = LabelEncoder()
    le_hotwaterheating = LabelEncoder()
    le_airconditioning = LabelEncoder()
    le_prefarea = LabelEncoder()
    le_furnishingstatus = LabelEncoder()
    
    df['mainroad'] = le_mainroad.fit_transform(df['mainroad'])
    df['guestroom'] = le_guestroom.fit_transform(df['guestroom'])
    df['basement'] = le_basement.fit_transform(df['basement'])
    df['hotwaterheating'] = le_hotwaterheating.fit_transform(df['hotwaterheating'])
    df['airconditioning'] = le_airconditioning.fit_transform(df['airconditioning'])
    df['prefarea'] = le_prefarea.fit_transform(df['prefarea'])
    df['furnishingstatus'] = le_furnishingstatus.fit_transform(df['furnishingstatus'])
    
    # Model train karo
    X = df.drop('price', axis=1)
    y = df['price']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Encoders dictionary mein save karo
    encoders = {
        'mainroad': le_mainroad,
        'guestroom': le_guestroom,
        'basement': le_basement,
        'hotwaterheating': le_hotwaterheating,
        'airconditioning': le_airconditioning,
        'prefarea': le_prefarea,
        'furnishingstatus': le_furnishingstatus
    }
    
    return model, encoders

# Load model and encoders
model, encoders = load_and_train_model()

# Get unique values from encoders for dropdowns
mainroad_options = list(encoders['mainroad'].classes_)
guestroom_options = list(encoders['guestroom'].classes_)
basement_options = list(encoders['basement'].classes_)
hotwaterheating_options = list(encoders['hotwaterheating'].classes_)
airconditioning_options = list(encoders['airconditioning'].classes_)
prefarea_options = list(encoders['prefarea'].classes_)
furnishingstatus_options = list(encoders['furnishingstatus'].classes_)


st.divider()
st.markdown("#### Enter the Details Below")

area = st.number_input("Enter Area (sq ft):", min_value=1000, max_value=20000, value=3000)
bedrooms = st.number_input("Enter Number of Bedrooms:", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Enter Number of Bathrooms:", min_value=1, max_value=10, value=2)
stories = st.number_input("Enter Number of Stories:", min_value=1, max_value=5, value=2)
mainroad = st.selectbox("Main Road Access:", mainroad_options)
guestroom = st.selectbox("Guest Room:", guestroom_options)
basement = st.selectbox("Basement:", basement_options)
hotwaterheating = st.selectbox("Hot Water Heating:", hotwaterheating_options)
airconditioning = st.selectbox("Air Conditioning:", airconditioning_options)
parking = st.number_input("Parking Spaces:", min_value=0, max_value=5, value=2)
prefarea = st.selectbox("Preferred Area:", prefarea_options)
furnishingstatus = st.selectbox("Furnishing Status:", furnishingstatus_options)

if st.button("Predict Price"):
    sample_data = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus
    }
    
    sample_df = pd.DataFrame([sample_data])
    
    # Transform using cached Label Encoders
    sample_df["mainroad"] = encoders['mainroad'].transform(sample_df["mainroad"])
    sample_df["guestroom"] = encoders['guestroom'].transform(sample_df["guestroom"])
    sample_df["basement"] = encoders['basement'].transform(sample_df["basement"])
    sample_df["hotwaterheating"] = encoders['hotwaterheating'].transform(sample_df["hotwaterheating"])
    sample_df["airconditioning"] = encoders['airconditioning'].transform(sample_df["airconditioning"])
    sample_df["prefarea"] = encoders['prefarea'].transform(sample_df["prefarea"])
    sample_df["furnishingstatus"] = encoders['furnishingstatus'].transform(sample_df["furnishingstatus"])
    
    # Predict
    prediction = model.predict(sample_df)
    
    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")

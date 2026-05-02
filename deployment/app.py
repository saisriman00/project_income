import streamlit as st
import joblib
import pandas as pd

with open("artifacts/best_model.txt") as f:
    best_model_name = f.read().strip()

model = joblib.load(f"artifacts/models/{best_model_name}.pkl")
preprocessor = joblib.load("artifacts/preprocessor.pkl")
label_encoder = joblib.load("artifacts/label_encoder.pkl")
feature_columns = joblib.load("artifacts/feature_columns.pkl")

st.title("Income Prediction")

age = st.number_input("Age", 18, 100)
hours = st.number_input("Hours per week", 1, 100)

if st.button("Predict"):
    input_dict = {col: 0 for col in feature_columns}
    input_dict["age"] = age
    input_dict["hours-per-week"] = hours

    input_df = pd.DataFrame([input_dict])
    X = preprocessor.transform(input_df)

    pred = model.predict(X)
    result = label_encoder.inverse_transform(pred)[0]

    st.write(result)

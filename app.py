import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
from tensorflow.keras.models import load_model

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Prediksi Gallstone",
    layout="wide"
)

# ===============================
# LOAD FILE
# ===============================
@st.cache_resource
def load_pipeline():
    return joblib.load("best_ann_pipeline.pkl")

@st.cache_resource
def load_model_ann():
    return load_model("best_ann_model.h5")

@st.cache_data
def load_best_params():
    with open("best_params.json", "r") as f:
        return json.load(f)

pipeline = load_pipeline()
model = load_model_ann()
best_params = load_best_params()

# ===============================
# HEADER
# ===============================
st.title("Aplikasi Prediksi Gallstone")
st.write("Prototype sistem pendukung keputusan berbasis ANN")

# ===============================
# INPUT
# ===============================
st.subheader("Input Data Pasien")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 45)
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
    glucose = st.number_input("Glucose", 50.0, 300.0, 100.0)
    hdl = st.number_input("HDL", 10.0, 100.0, 45.0)

with col2:
    ldl = st.number_input("LDL", 50.0, 300.0, 120.0)
    trig = st.number_input("Triglyceride", 50.0, 500.0, 150.0)
    ast = st.number_input("AST", 5.0, 200.0, 25.0)
    alt = st.number_input("ALT", 5.0, 200.0, 30.0)

# ===============================
# PREDICTION
# ===============================
if st.button("Prediksi"):
    input_df = pd.DataFrame([{
        "Age": age,
        "BMI": bmi,
        "Glucose": glucose,
        "HDL": hdl,
        "LDL": ldl,
        "Triglyceride": trig,
        "AST": ast,
        "ALT": alt
    }])

    X_scaled = pipeline.transform(input_df)
    prob = model.predict(X_scaled)[0][0]

    st.subheader("Hasil Prediksi")
    st.write(f"Probabilitas Gallstone: **{prob:.3f}**")

    if prob >= 0.5:
        st.error("Terindikasi Gallstone")
    else:
        st.success("Tidak Terindikasi Gallstone")

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("Parameter Model Terbaik")
st.sidebar.json(best_params)

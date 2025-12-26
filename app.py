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
# LOAD ARTEFAK INFERENSI (INI PERUBAHAN UTAMA)
# ===============================
@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_resource
def load_model_ann():
    return load_model("best_ann_model.h5")

@st.cache_data
def load_feature_names():
    with open("feature_names.json") as f:
        return json.load(f)

@st.cache_data
def load_best_params():
    with open("best_params.json") as f:
        return json.load(f)

scaler = load_scaler()
model = load_model_ann()
feature_names = load_feature_names()
best_params = load_best_params()

# ===============================
# HEADER
# ===============================
st.title("Aplikasi Prediksi Gallstone")
st.write("Prototype sistem pendukung keputusan berbasis ANN")

# ===============================
# INPUT (contoh sebagian fitur)
# ===============================
st.subheader("Input Data Pasien")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", 18, 100, 45)
    BMI = st.number_input("BMI", 10.0, 50.0, 25.0)
    Glucose = st.number_input("Glucose", 50.0, 300.0, 100.0)
    HDL = st.number_input("HDL", 10.0, 100.0, 45.0)

with col2:
    LDL = st.number_input("LDL", 50.0, 300.0, 120.0)
    Triglyceride = st.number_input("Triglyceride", 50.0, 500.0, 150.0)
    AST = st.number_input("AST", 5.0, 200.0, 25.0)
    ALT = st.number_input("ALT", 5.0, 200.0, 30.0)

# ===============================
# PREDIKSI
# ===============================
if st.button("Prediksi"):
    # 1️⃣ Buat input awal
    input_df = pd.DataFrame([{
        "Age": Age,
        "BMI": BMI,
        "Glucose": Glucose,
        "HDL": HDL,
        "LDL": LDL,
        "Triglyceride": Triglyceride,
        "AST": AST,
        "ALT": ALT
    }])

    # 2️⃣ Lengkapi ke 30 fitur (ISI DEFAULT = MEAN)
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0  # atau mean klinis

    # 3️⃣ URUTAN WAJIB SAMA
    input_df = input_df[feature_names]

    # 4️⃣ Scaling
    X_scaled = scaler.transform(input_df)

    # 5️⃣ Prediksi
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

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
from tensorflow.keras.models import load_model

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Prediksi Gallstone",
    layout="wide"
)

# ======================================================
# LOAD ARTEFAK INFERENSI
# ======================================================
@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_resource
def load_imputer():
    return joblib.load("knn_imputer.pkl")

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
imputer = load_imputer()
model = load_model_ann()
feature_names = load_feature_names()
best_params = load_best_params()

# ======================================================
# HEADER
# ======================================================
st.title("Aplikasi Prediksi Gallstone")
st.markdown(
    """
    **Sistem Pendukung Keputusan Berbasis Artificial Neural Network (ANN)**  

    - Model menggunakan **30 fitur**
    - Pengguna **tidak wajib mengisi seluruh fitur**
    - Fitur yang tidak diisi akan **diimputasi otomatis menggunakan KNN Imputation**
    """
)

# ======================================================
# INPUT DATA PASIEN (FLEKSIBEL)
# ======================================================
st.subheader("Input Data Pasien (Isi sesuai ketersediaan data)")

col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input("Age", min_value=0, max_value=120, value=45)
    BMI = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0)
    Weight = st.number_input("Weight (kg)", min_value=0.0, max_value=200.0, value=70.0)
    Height = st.number_input("Height (cm)", min_value=0.0, max_value=220.0, value=165.0)

with col2:
    Glucose = st.number_input("Glucose", min_value=0.0, max_value=500.0, value=100.0)
    HDL = st.number_input("HDL", min_value=0.0, max_value=150.0, value=45.0)
    LDL = st.number_input("LDL", min_value=0.0, max_value=300.0, value=120.0)
    Triglyceride = st.number_input("Triglyceride", min_value=0.0, max_value=500.0, value=150.0)

with col3:
    AST = st.number_input("AST", min_value=0.0, max_value=300.0, value=25.0)
    ALT = st.number_input("ALT", min_value=0.0, max_value=300.0, value=30.0)
    Creatinine = st.number_input("Creatinine", min_value=0.0, max_value=10.0, value=1.0)
    GFR = st.number_input("GFR", min_value=0.0, max_value=200.0, value=90.0)

# ======================================================
# PREDIKSI
# ======================================================
if st.button("Prediksi"):
    # --------------------------------------------
    # 1. Data input user (SEBAGIAN FITUR)
    # --------------------------------------------
    input_data = {
        "Age": Age,
        "BMI": BMI,
        "Weight": Weight,
        "Height": Height,
        "Glucose": Glucose,
        "HDL": HDL,
        "LDL": LDL,
        "Triglyceride": Triglyceride,
        "AST": AST,
        "ALT": ALT,
        "Creatinine": Creatinine,
        "GFR": GFR,
    }

    input_df = pd.DataFrame([input_data])

    # --------------------------------------------
    # 2. Lengkapi ke 30 fitur (missing = NaN)
    # --------------------------------------------
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = np.nan

    # Urutan HARUS sama seperti training
    input_df = input_df[feature_names]

    # --------------------------------------------
    # 3. KNN Imputation (TANPA FIT ULANG)
    # --------------------------------------------
    X_imputed = imputer.transform(input_df)

    # --------------------------------------------
    # 4. Scaling
    # --------------------------------------------
    X_scaled = scaler.transform(X_imputed)

    # --------------------------------------------
    # 5. Prediksi ANN
    # --------------------------------------------
    prob = model.predict(X_scaled)[0][0]

    # ==================================================
    # OUTPUT
    # ==================================================
    st.subheader("Hasil Prediksi")
    st.metric("Probabilitas Gallstone", f"{prob:.3f}")

    if prob < 0.3:
        st.success("Risiko Rendah")
    elif prob < 0.7:
        st.warning("Risiko Sedang")
    else:
        st.error("Risiko Tinggi")

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header("Informasi Model")
st.sidebar.json(best_params)

st.sidebar.info(
    "Catatan:\n"
    "- Model menggunakan 30 fitur\n"
    "- Input pengguna bersifat fleksibel\n"
    "- Nilai yang tidak diinput diimputasi dengan KNN\n"
    "- Model tidak melakukan retraining saat inferensi"
)

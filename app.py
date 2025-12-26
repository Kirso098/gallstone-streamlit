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
    **Sistem SKrinning Berbasis Artificial Neural Network (ANN)**  

    - Model menggunakan **30 fitur input**
    - Pengguna **tidak wajib mengisi seluruh fitur**
    - **Fitur turunan dihitung otomatis**
    - Fitur yang belum tersedia **diimputasi menggunakan KNN Imputation**
    """
)

# ======================================================
# INPUT DATA PASIEN (FITUR PRIMER)
# ======================================================
st.subheader("Input Data Pasien (Fitur Primer)")

col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input("Age", 0, 120, 45)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Height = st.number_input("Height (cm)", 0.0, 250.0, 165.0)
    Weight = st.number_input("Weight (kg)", 0.0, 200.0, 70.0)

with col2:
    Glucose = st.number_input("Glucose", 0.0, 500.0, 100.0)
    ALT = st.number_input("Alanin Aminotransferaz (ALT)", 0.0, 300.0, 30.0)
    AST = st.number_input("Aspartat Aminotransferaz (AST)", 0.0, 300.0, 25.0)
    Creatinine = st.number_input("Creatinine", 0.0, 10.0, 1.0)
    GFR = st.number_input("Glomerular Filtration Rate (GFR)", 0.0, 200.0, 90.0)

with col3:
    HDL = st.number_input("High Density Lipoprotein (HDL)", 0.0, 150.0, 45.0)
    LDL = st.number_input("Low Density Lipoprotein (LDL)", 0.0, 300.0, 120.0)
    Triglyceride = st.number_input("Triglyceride", 0.0, 500.0, 150.0)
    TC = st.number_input("Total Cholesterol (TC)", 0.0, 400.0, 200.0)
    HGB = st.number_input("Hemoglobin (HGB)", 0.0, 25.0, 14.0)

# ======================================================
# FITUR TURUNAN (AUTO COMPUTE)
# ======================================================
st.subheader("Fitur Turunan (Dihitung Otomatis)")

eps = 1e-6  # untuk menghindari pembagian nol

BMI = Weight / ((Height / 100) ** 2 + eps)
TC_HDL = TC / (HDL + eps)
LDL_HDL = LDL / (HDL + eps)
TG_HDL = Triglyceride / (HDL + eps)
AIP = np.log10((Triglyceride + eps) / (HDL + eps))
De_Ritis = AST / (ALT + eps)

col4, col5 = st.columns(2)

with col4:
    st.text_input("Body Mass Index (BMI)", f"{BMI:.2f}", disabled=True)
    st.text_input("TC/HDL Ratio", f"{TC_HDL:.2f}", disabled=True)
    st.text_input("LDL/HDL Ratio", f"{LDL_HDL:.2f}", disabled=True)
    st.text_input("Triglyceride/HDL Ratio", f"{TG_HDL:.2f}", disabled=True)

with col5:
    st.text_input("Atherogenic Index", f"{AIP:.2f}", disabled=True)
    st.text_input("De Ritis Ratio", f"{De_Ritis:.2f}", disabled=True)

# ======================================================
# PREDIKSI
# ======================================================
if st.button("Prediksi"):
    # --------------------------------------------
    # 1. Gabungkan fitur primer + turunan
    # --------------------------------------------
    input_data = {
        "Age": Age,
        "Gender": 1 if Gender == "Male" else 0,
        "Height": Height,
        "Weight": Weight,
        "Glucose": Glucose,
        "Alanin Aminotransferaz (ALT)": ALT,
        "Aspartat Aminotransferaz (AST)": AST,
        "Creatinine": Creatinine,
        "Glomerular Filtration Rate (GFR)": GFR,
        "High Density Lipoprotein (HDL)": HDL,
        "Low Density Lipoprotein (LDL)": LDL,
        "Triglyceride": Triglyceride,
        "Total Cholesterol (TC)": TC,
        "Hemoglobin (HGB)": HGB,
        "Body Mass Index (BMI)": BMI,
        "TC/HDL Ratio": TC_HDL,
        "LDL/HDL Ratio": LDL_HDL,
        "Triglyceride/HDL Ratio": TG_HDL,
        "Atherogenic Index": AIP,
        "De Ritis Ratio": De_Ritis,
    }

    input_df = pd.DataFrame([input_data])

    # --------------------------------------------
    # 2. Lengkapi ke 30 fitur (missing → NaN)
    # --------------------------------------------
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = np.nan

    # Urutan kolom harus sama
    input_df = input_df[feature_names]

    # --------------------------------------------
    # 3. KNN Imputation → Scaling → Predict
    # --------------------------------------------
    X_imputed = imputer.transform(input_df)
    X_scaled = scaler.transform(X_imputed)
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
    "- Model ANN menggunakan 30 fitur\n"
    "- Fitur turunan dihitung otomatis oleh sistem\n"
    "- Fitur kosong diimputasi menggunakan KNN\n"
    "- Tidak ada retraining saat inferensi"
)

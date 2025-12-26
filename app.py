import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
from tensorflow.keras.models import load_model

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="Prediksi Gallstone", layout="wide")

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

    - Model menggunakan **30 fitur input**
    - Pengguna **tidak wajib mengisi seluruh fitur**
    - **Fitur turunan & agregat dihitung otomatis**
    - Nilai kosong **diimputasi menggunakan KNN Imputation**
    """
)

# ======================================================
# INPUT DATA PASIEN – FITUR PRIMER
# ======================================================
st.subheader("Input Data Pasien (Fitur Primer)")

col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input("Age", min_value=0, max_value=120, value=None)
    Gender = st.selectbox("Gender", ["", "Male", "Female"])
    Height = st.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=None)
    Weight = st.number_input("Weight (kg)", min_value=0.0, max_value=200.0, value=None)

with col2:
    Glucose = st.number_input("Glucose", min_value=0.0, max_value=500.0, value=None)
    ALT = st.number_input("Alanin Aminotransferaz (ALT)", min_value=0.0, max_value=300.0, value=None)
    AST = st.number_input("Aspartat Aminotransferaz (AST)", min_value=0.0, max_value=300.0, value=None)
    Creatinine = st.number_input("Creatinine", min_value=0.0, max_value=10.0, value=None)
    GFR = st.number_input("Glomerular Filtration Rate (GFR)", min_value=0.0, max_value=200.0, value=None)

with col3:
    HDL = st.number_input("High Density Lipoprotein (HDL)", min_value=0.0, max_value=150.0, value=None)
    LDL = st.number_input("Low Density Lipoprotein (LDL)", min_value=0.0, max_value=300.0, value=None)
    Triglyceride = st.number_input("Triglyceride", min_value=0.0, max_value=500.0, value=None)
    TC = st.number_input("Total Cholesterol (TC)", min_value=0.0, max_value=400.0, value=None)
    HGB = st.number_input("Hemoglobin (HGB)", min_value=0.0, max_value=25.0, value=None)

# ======================================================
# KOMORBID (CHECKBOX)
# ======================================================
st.subheader("Komorbiditas")

c1, c2 = st.columns(2)
with c1:
    DM = st.checkbox("Diabetes Mellitus (DM)")
    CAD = st.checkbox("Coronary Artery Disease (CAD)")
with c2:
    Hyperlipidemia = st.checkbox("Hyperlipidemia")
    Hypothyroidism = st.checkbox("Hypothyroidism")

# ======================================================
# INPUT KOMPOSISI TUBUH (TAMBAHAN OPSI A)
# ======================================================
st.subheader("Komposisi Tubuh (Opsional)")

c3, c4 = st.columns(2)
with c3:
    LeanMass = st.number_input("Lean Mass (LM) (%)", min_value=0.0, max_value=100.0, value=None)
    TBFR = st.number_input("Total Body Fat Ratio (TBFR) (%)", min_value=0.0, max_value=100.0, value=None)

with c4:
    TBW = st.number_input("Total Body Water (TBW)", min_value=0.0, max_value=100.0, value=None)

# ======================================================
# HITUNG FITUR TURUNAN & AGREGAT
# ======================================================
eps = 1e-6

BMI = (Weight / ((Height / 100) ** 2 + eps)) if Weight and Height else np.nan
TC_HDL = (TC / (HDL + eps)) if TC and HDL else np.nan
LDL_HDL = (LDL / (HDL + eps)) if LDL and HDL else np.nan
TG_HDL = (Triglyceride / (HDL + eps)) if Triglyceride and HDL else np.nan
AIP = np.log10((Triglyceride + eps) / (HDL + eps)) if Triglyceride and HDL else np.nan
DeRitis = (AST / (ALT + eps)) if AST and ALT else np.nan

NLM = (Weight * (1 - LeanMass / 100)) if Weight and LeanMass else np.nan
BF_Water = (TBFR / (TBW + eps)) if TBFR and TBW else np.nan

Comorbidity = sum([
    int(DM), int(CAD), int(Hyperlipidemia), int(Hypothyroidism)
])

# ======================================================
# TAMPILKAN FITUR TURUNAN (READ-ONLY)
# ======================================================
st.subheader("Fitur Turunan & Agregat (Otomatis)")

d1, d2, d3 = st.columns(3)

with d1:
    st.text_input("Body Mass Index (BMI)", "" if np.isnan(BMI) else f"{BMI:.2f}", disabled=True)
    st.text_input("TC/HDL Ratio", "" if np.isnan(TC_HDL) else f"{TC_HDL:.2f}", disabled=True)
    st.text_input("LDL/HDL Ratio", "" if np.isnan(LDL_HDL) else f"{LDL_HDL:.2f}", disabled=True)

with d2:
    st.text_input("Triglyceride/HDL Ratio", "" if np.isnan(TG_HDL) else f"{TG_HDL:.2f}", disabled=True)
    st.text_input("Atherogenic Index", "" if np.isnan(AIP) else f"{AIP:.2f}", disabled=True)
    st.text_input("De Ritis Ratio", "" if np.isnan(DeRitis) else f"{DeRitis:.2f}", disabled=True)

with d3:
    st.text_input("Non-Lean Mass (NLM)", "" if np.isnan(NLM) else f"{NLM:.2f}", disabled=True)
    st.text_input("Body Fat/Water Ratio", "" if np.isnan(BF_Water) else f"{BF_Water:.2f}", disabled=True)
    st.text_input("Comorbidity Score", f"{Comorbidity}", disabled=True)

# ======================================================
# PREDIKSI
# ======================================================
if st.button("Prediksi"):
    input_data = {
        "Age": Age,
        "Gender": 1 if Gender == "Male" else 0 if Gender == "Female" else np.nan,
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
        "Diabetes Mellitus (DM)": int(DM),
        "Coronary Artery Disease (CAD)": int(CAD),
        "Hyperlipidemia": int(Hyperlipidemia),
        "Hypothyroidism": int(Hypothyroidism),
        "Lean Mass (LM) (%)": LeanMass,
        "Total Body Fat Ratio (TBFR) (%)": TBFR,
        "Total Body Water (TBW)": TBW,
        "Body Mass Index (BMI)": BMI,
        "TC/HDL Ratio": TC_HDL,
        "LDL/HDL Ratio": LDL_HDL,
        "Atherogenic Index": AIP,
        "Triglyceride/HDL Ratio": TG_HDL,
        "Non-Lean Mass (NLM)": NLM,
        "Body Fat/Water Ratio": BF_Water,
        "De Ritis Ratio": DeRitis,
        "Comorbidity": Comorbidity,
    }

    input_df = pd.DataFrame([input_data])

    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = np.nan

    input_df = input_df[feature_names]

    X_imputed = imputer.transform(input_df)
    X_scaled = scaler.transform(X_imputed)
    prob = model.predict(X_scaled)[0][0]

    st.subheader("Hasil Prediksi")
    st.metric("Probabilitas Gallstone", f"{prob:.3f}")

    if prob >= 0.5:
        st.markdown("### 🩺 Status Kesehatan: **Gallstone**")
    else:
        st.markdown("### 🟢 Status Kesehatan: **Healthy**")

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

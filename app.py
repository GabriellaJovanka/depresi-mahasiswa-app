import streamlit as st
import pandas as pd
import joblib
import os

# 1. Load Model
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, 'model_final_depresi.pkl')
    return joblib.load(model_path)

model = load_model()

st.set_page_config(page_title="Prediksi Depresi Mahasiswa", layout="wide")
st.title("🎓 Aplikasi Prediksi Tingkat Depresi Mahasiswa")
st.write("Isi semua data di bawah ini untuk mendapatkan hasil analisis.")

# --- FORM INPUT ---
with st.form("main_form"):
    # Kita bagi 3 kolom agar tidak terlalu panjang ke bawah
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=60, value=20)
        city = st.text_input("City", "Jakarta")
        profession = st.text_input("Profession", "Student")
        academic_pressure = st.slider("Academic Pressure (1-5)", 1, 5, 3)
        work_pressure = st.slider("Work Pressure (1-5)", 1, 5, 1)

    with col2:
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=4.0, value=3.5, step=0.01)
        study_satisfaction = st.slider("Study Satisfaction (1-5)", 1, 5, 3)
        job_satisfaction = st.slider("Job Satisfaction (1-5)", 1, 5, 1)
        sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
        dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])

    with col3:
        degree = st.text_input("Degree", "S1")
        suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts ?", ["Yes", "No"])
        work_study_hours = st.number_input("Work/Study Hours", min_value=0, max_value=24, value=8)
        financial_stress = st.slider("Financial Stress (1-5)", 1, 5, 3)
        family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

    submitted = st.form_submit_button("Analisis Sekarang")

# --- PROSES PREDIKSI ---
if submitted:
    try:
        # MENYUSUN 16 KOLOM SESUAI URUTAN YANG DIMINTA MODEL
        input_dict = {
            'Gender': gender,
            'Age': age,
            'City': city,
            'Profession': profession,
            'Academic Pressure': academic_pressure,
            'Work Pressure': work_pressure,
            'CGPA': cgpa,
            'Study Satisfaction': study_satisfaction,
            'Job Satisfaction': job_satisfaction,
            'Sleep Duration': sleep_duration,
            'Dietary Habits': dietary_habits,
            'Degree': degree,
            'Have you ever had suicidal thoughts ?': suicidal_thoughts,
            'Work/Study Hours': work_study_hours,
            'Financial Stress': financial_stress,
            'Family History of Mental Illness': family_history
        }
        
        # Konversi ke DataFrame
        data_df = pd.DataFrame([input_dict])

        # Prediksi
        prediction = model.predict(data_df)
        probability = model.predict_proba(data_df)[0][1]

        # Tampilan Hasil
        st.divider()
        if prediction[0] == 1:
            st.error(f"⚠️ Hasil: Berisiko Depresi (Tingkat Risiko: {probability:.2%})")
        else:
            st.success(f"✅ Hasil: Tidak Berisiko (Tingkat Aman: {1-probability:.2%})")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")

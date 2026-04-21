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

st.title("Aplikasi Analisis Kesehatan Mental Mahasiswa")
st.write("Silakan isi formulir di bawah ini untuk melakukan prediksi.")

# --- FORM INPUT ---
with st.form("main_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=60, value=20)
        gender = st.selectbox("Gender", ["Male", "Female"])
        academic_pressure = st.slider("Academic Pressure (1-5)", 1, 5, 3)
        study_satisfaction = st.slider("Study Satisfaction (1-5)", 1, 5, 3)
        cgpa = st.number_input("CGPA / IPK", min_value=0.0, max_value=4.0, value=3.5, step=0.01)
        
    with col2:
        # Kolom baru yang kamu minta:
        work_study_hours = st.number_input("Work/Study Hours (per day)", min_value=0, max_value=24, value=8)
        financial_stress = st.slider("Financial Stress (1-5)", 1, 5, 3)
        suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])
        
        # Tambahan fitur lain jika ada (sesuaikan dengan datasetmu)
        sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
        dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])

    submitted = st.form_submit_button("Analisis Sekarang")

# --- PROSES PREDIKSI ---
# --- BAGIAN PROSES PREDIKSI ---
if submitted:
    try:
        # Urutan ini disesuaikan PERSIS dengan output Google Colab kamu
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
            'Have you ever had suicidal thoughts ?': suicidal_thoughts, # Perhatikan spasi sebelum '?'
            'Work/Study Hours': work_study_hours,
            'Financial Stress': financial_stress,
            'Family History of Mental Illness': family_history
        }
        
        # Ubah ke DataFrame
        data_df = pd.DataFrame([input_dict])

        # Lakukan Prediksi
        prediction = model.predict(data_df)
        probability = model.predict_proba(data_df)[0][1]

        # Tampilkan Hasil
        st.divider()
        if prediction[0] == 1:
            st.error(f"⚠️ Hasil Prediksi: Berisiko Depresi ({probability:.2%})")
        else:
            st.success(f"✅ Hasil Prediksi: Tidak Berisiko ({1-probability:.2%})")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

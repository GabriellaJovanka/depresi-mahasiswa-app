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

st.title("Aplikasi Prediksi Depresi Mahasiswa")

# --- FORM INPUT ---
with st.form("main_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        # Pastikan semua variabel ini ADA dan DEFINED
        age = st.number_input("Age", min_value=18, max_value=60, value=20)
        gender = st.selectbox("Gender", ["Male", "Female"])
        academic_pressure = st.slider("Academic Pressure (1-5)", 1, 5, 3)
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=4.0, value=3.5, step=0.01)
        study_satisfaction = st.slider("Study Satisfaction (1-5)", 1, 5, 3)

    with col2:
        work_study_hours = st.number_input("Work/Study Hours", min_value=0, max_value=24, value=8)
        financial_stress = st.slider("Financial Stress (1-5)", 1, 5, 3)
        suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts ?", ["Yes", "No"])
        sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
        dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])

    submitted = st.form_submit_button("Prediksi")

# --- PROSES PREDIKSI ---
if submitted:
    try:
        # Sesuai error sebelumnya, modelmu HANYA minta 10 kolom.
        # Jadi kita HANYA masukkan 10 variabel yang sudah dibuat di atas.
        # JANGAN masukkan 'city', 'profession', atau 'degree' jika tidak ada inputnya.
        
        input_dict = {
            'Gender': gender,
            'Age': age,
            'Academic Pressure': academic_pressure,
            'CGPA': cgpa,
            'Study Satisfaction': study_satisfaction,
            'Sleep Duration': sleep_duration,
            'Dietary Habits': dietary_habits,
            'Have you ever had suicidal thoughts ?': suicidal_thoughts,
            'Work/Study Hours': work_study_hours,
            'Financial Stress': financial_stress
        }
        
        data_df = pd.DataFrame([input_dict])

        # Prediksi
        prediction = model.predict(data_df)
        
        st.divider()
        if prediction[0] == 1:
            st.error("⚠️ Hasil: Berisiko Depresi")
        else:
            st.success("✅ Hasil: Tidak Berisiko")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        # Jika masih error jumlah kolom, aktifkan baris di bawah ini untuk cek:
        # st.write(model.feature_names_in_)

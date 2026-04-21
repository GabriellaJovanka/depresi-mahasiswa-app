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
if submitted:
    try:
        # PENTING: Susun urutan kolom di bawah ini AGAR SAMA PERSIS dengan dataset Colab
        # Pastikan nama key (sebelah kiri) SAMA PERSIS dengan nama kolom di CSV
        
        input_dict = {
            'Age': age,
            'Gender': gender,
            'Academic Pressure': academic_pressure,
            'Work/Study Hours': work_study_hours,     # Kolom baru
            'Financial Stress': financial_stress,     # Kolom baru
            'Study Satisfaction': study_satisfaction,
            'Sleep Duration': sleep_duration,
            'Dietary Habits': dietary_habits,
            'Have you ever had suicidal thoughts?': suicidal_thoughts, # Kolom baru
            'CGPA': cgpa
        }
        
        # Mengubah ke DataFrame
        data_df = pd.DataFrame([input_dict])

        # Lakukan Prediksi
        prediction = model.predict(data_df)
        probability = model.predict_proba(data_df)[0][1]

        # Tampilkan Hasil
        st.divider()
        if prediction[0] == 1:
            st.error(f"⚠️ Hasil Prediksi: Berisiko Depresi (Probabilitas: {probability:.2%})")
            st.warning("Saran: Segera konsultasikan kondisi Anda dengan profesional atau layanan kesehatan mental terdekat.")
        else:
            st.success(f"✅ Hasil Prediksi: Tidak Berisiko (Probabilitas: {1-probability:.2%})")
            st.info("Saran: Tetap jaga pola hidup sehat dan manajemen stres yang baik.")

    except Exception as e:
        st.error(f"Terjadi kesalahan teknis: {e}")
        st.info("💡 Tips: Periksa kembali apakah urutan dan nama kolom di app.py sudah sama dengan model di Colab.")

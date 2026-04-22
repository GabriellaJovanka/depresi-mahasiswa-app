# --- Deteksi Tingkat Depresi pada Mahasiswa Menggunakan Machine Learning ---

# --- Gabriella Jovanka Bustan (A11.2023.14861) ---

import streamlit as st
import pandas as pd
import joblib
import os

# --- LOADING MODEL ---
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, 'model_final_depresi.pkl')
    return joblib.load(model_path)

model_pipeline = load_model()

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Depresi Mahasiswa", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        /* Background Utama */
        .stApp {
            background-color: #000000; 
        }

        /* Warna Teks Judul Halaman */
        .stApp h1, .stApp h2 {
            color: #FFFFFF !important;
        }

        /* Styling Form */
        div[data-testid="stForm"] {
            background-color: #355872;
            padding: 30px;
            border-radius: 15px;
            border: none;
            box-shadow: 0px 10px 25px rgba(0,0,0,0.1);
        }

        /* Warna Teks di Dalam Form */
        div[data-testid="stForm"] label, 
        div[data-testid="stForm"] p, 
        div[data-testid="stForm"] h3 {
            color: white !important;
        }

        /* TOMBOL ANALISIS */
        div[data-testid="stFormSubmitButton"] button {
            background-color: #000000 !important;
            border: 2px solid #FFFFFF !important;
            font-weight: bold;
            font-size: 18px;
            padding: 10px 0px;
            border-radius: 10px;
            transition: 0.3s;
        }

        div[data-testid="stFormSubmitButton"] button:hover {
            background-color: #FFFFF !important;
            border-color: #FFFFFF !important;
            color: #000000 !important;
            transform: scale(1.01);
        }

        /* Card Hasil Prediksi */
        .result-card {
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin-top: 20px;
        }
        
        .result-card h2, .result-card p, .result-card div, .result-card h3 {
            color: white !important;
        }

        .prob-text {
            font-size: 60px;
            font-weight: 900;
            margin: 10px 0;
        }

        /* Force warna hitam pada kotak notifikasi */
        [data-testid="stNotification"] p, 
        [data-testid="stNotification"] strong {
            color: #000000 !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 Aplikasi Prediksi Tingkat Depresi Mahasiswa")

# --- FORM INPUT ---
with st.form("main_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=60, value=20)
        city = st.text_input("City", "Jakarta")
        profession = st.text_input("Profession", "Student")
        academic_pressure = st.slider("Academic Pressure (0-5)", 0, 5, 3)
        work_pressure = st.slider("Work Pressure (0-5)", 0, 5, 0)
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.01)
        study_satisfaction = st.slider("Study Satisfaction (0-5)", 0, 5, 3)
    
    with col2:
        job_satisfaction = st.slider("Job Satisfaction (0-5)", 0, 5, 0)
        sleep_duration = st.selectbox("Sleep Duration", ["Less Than 5 Hours", "5-6 Hours", "7-8 Hours", "More Than 8 Hours"])
        dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
        degree = st.text_input("Degree", "S1")
        suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts ?", ["Yes", "No"])
        work_study_hours = st.number_input("Work/Study Hours", min_value=0, max_value=24, value=8)
        financial_stress = st.slider("Financial Stress (1-5)", 1, 5, 3)
        family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

    st.markdown("---")

    # Tombol 
    submitted = st.form_submit_button("Analisis Sekarang", use_container_width=True)

# --- PROSES PREDIKSI ---
if submitted:
    try:
        input_dict = {
            'Gender': gender, 'Age': age, 'City': city, 'Profession': profession,
            'Academic Pressure': academic_pressure, 'Work Pressure': work_pressure,
            'CGPA': cgpa, 'Study Satisfaction': study_satisfaction,
            'Job Satisfaction': job_satisfaction, 'Sleep Duration': sleep_duration,
            'Dietary Habits': dietary_habits, 'Degree': degree,
            'Have you ever had suicidal thoughts ?': suicidal_thoughts,
            'Work/Study Hours': work_study_hours, 'Financial Stress': financial_stress,
            'Family History of Mental Illness': family_history
        }
        
        data_df = pd.DataFrame([input_dict])
        
        # Pre-processing 
        for col in data_df.select_dtypes(include=['object']).columns:
            data_df[col] = data_df[col].astype(str).str.strip()

        # Prediksi Label dan Probabilitas
        prediction = model_pipeline.predict(data_df)[0]
        # Pastikan menggunakan model_pipeline jika predict_proba ada di dalam pipeline
        probability = model_pipeline.predict_proba(data_df)[0][1] * 100
        
        st.divider()

        # Logika tampilan output
        if prediction == 1:
            bg_color = "#355872"
            status_text = "Berisiko Tinggi Depresi"
            icon = "⚠️"
        else:
            bg_color = "#27AE60"
            status_text = "Risiko Rendah / Tidak Berisiko"
            icon = "✅"

        st.markdown(f"""
            <div class="result-card" style="background-color: {bg_color}; border: 5px solid rgba(255,255,255, 0.1);">
                <h2 style="margin-top: 0;">{icon} Hasil Analisis {icon}</h2>
                <p style="margin-bottom: 0; opacity: 0.9;">Probabilitas:</p>
                <div class="prob-text">{probability:.1f}%</div>
                <h3 style="margin-bottom: 0;">Status: {status_text}</h3>
            </div>
        """, unsafe_allow_html=True)

        if prediction == 1:
            st.warning("**Rekomendasi:** Hasil ini menunjukkan indikasi tekanan psikologis yang kuat. Jangan ragu untuk berbicara dengan konselor, psikolog, atau orang terdekat yang Anda percayai.")
        else:
            st.success("**Rekomendasi:** Pertahankan kesehatan mental Anda. Tetap luangkan waktu untuk istirahat dan hobi di tengah kesibukan akademik.")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")

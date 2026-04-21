import streamlit as st
import pandas as pd
import joblib
import os

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, 'model_final_depresi.pkl')
    return joblib.load(model_path)

model_pipeline = load_model()

st.set_page_config(page_title="Prediksi Depresi Mahasiswa", layout="wide")
st.title("🎓 Aplikasi Prediksi Tingkat Depresi Mahasiswa")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        /* Background Utama */
        .stApp {
            background-color: #EFD2B0; 
        }

        /* Warna Teks Judul Halaman */
        .stApp h1, .stApp h2 {
            color: #355872 !important;
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
        div[data-testid="stForm"] label, div[data-testid="stForm"] p, div[data-testid="stForm"] h3 {
            color: white !important;
        }

        /* TOMBOL ANALISIS */
        div[data-testid="stFormSubmitButton"] button {
            background-color: #EFD2B0 !important;
            color: #355872 !important;
            border: 2px solid #EFD2B0 !important;
            font-weight: bold;
            font-size: 18px;
            padding: 10px 0px;
            border-radius: 10px;
            transition: 0.3s;
        }

        div[data-testid="stFormSubmitButton"] button:hover {
            background-color: #f7e6d2 !important;
            border-color: #355872 !important;
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
        [data-testid="stNotification"] {
            color: #000000 !important;
        }
        [data-testid="stNotification"] div, 
        [data-testid="stNotification"] p, 
        [data-testid="stNotification"] strong {
            color: #000000 !important;
        }
    </style>
""", unsafe_allow_html=True)


with st.form("main_form"):
    col1, col2 = st.columns(2)
    
    with col1: # Pastikan ini menjorok ke dalam 'with st.form'
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=60, value=20)
        city = st.text_input("City", "Jakarta")
        profession = st.text_input("Profession", "Student")
        academic_pressure = st.slider("Academic Pressure (0-5)", 0, 5, 3)
        work_pressure = st.slider("Work Pressure (0-5)", 0, 5, 0)
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.01)
        study_satisfaction = st.slider("Study Satisfaction (0-5)", 0, 5, 3)
    
    with col2: # Pastikan ini sejajar dengan col1
        job_satisfaction = st.slider("Job Satisfaction (0-5)", 0, 5, 0)
        sleep_duration = st.selectbox("Sleep Duration", ["Less Than 5 Hours", "5-6 Hours", "7-8 Hours", "More Than 8 Hours"])
        dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
        degree = st.text_input("Degree", "S1")
        suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts ?", ["Yes", "No"])
        work_study_hours = st.number_input("Work/Study Hours", min_value=0, max_value=24, value=8)
        financial_stress = st.slider("Financial Stress (1-5)", 1, 5, 3)
        family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

    st.markdown("---")

    # 3. Javascript (DIBETULKAN: Ditambah penutup tag script)
    st.components.v1.html(
        """
        <script>
            var customBtn = window.parent.document.getElementById('custom-submit-btn');
            var realBtn = window.parent.document.querySelector('div.stButton > button[kind="primaryFormSubmit"]');
            if (customBtn && realBtn) {
                customBtn.onclick = function() {
                    realBtn.click();
                };
            }
        </script>
        """,
        height=0
    )

    # Tombol asli tetap harus ada di dalam form agar 'submitted' bekerja
    submitted = st.form_submit_button("Analisis Sekarang", use_container_width=True)

# --- PROSES PREDIKSI (Di luar blok with st.form) ---
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
        for col in data_df.select_dtypes(include=['object']).columns:
            data_df[col] = data_df[col].astype(str).str.strip()

        prediction = model_pipeline.predict(data_df)
        
        st.divider()
        if prediction[0] == 1:
            st.error("⚠️ Hasil: Berisiko Depresi")
        else:
            st.success("✅ Hasil: Tidak Berisiko")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")

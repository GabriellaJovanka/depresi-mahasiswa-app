import streamlit as st
import pandas as pd
import joblib
import os

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, 'model_final_depresi.pkl')
    return joblib.load(model_path)

model = load_model()

st.set_page_config(page_title="Prediksi Depresi Mahasiswa", layout="wide")
st.title("🔍 Aplikasi Prediksi Tingkat Depresi Mahasiswa")

# --- SET BACKGROUND BIRU NAVY ---
st.markdown("""
    <style>
        /* Mengubah background utama aplikasi */
        .stApp {
            background-color: #406AAF; /* Navy Blue Hex */
        }

        /* Opsional: Mengubah warna teks jadi putih agar terbaca di background gelap */
        .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp span, .stApp label {
            color: white !important;
        }

        /* Membuat kotak form tetap terlihat rapi dengan background agak transparan */
        div[data-testid="stForm"] {
            background-color: #355872;
            padding: 20px;
            border-radius: 10px;
        }

    </style>
""", unsafe_allow_html=True)

with st.form("main_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=60, value=20)
        city = st.text_input("City", "Jakarta")
        profession = st.text_input("Profession", "Student")
        academic_pressure = st.slider("Academic Pressure (1-5)", 1, 5, 3)
        work_pressure = st.slider("Work Pressure (1-5)", 1, 5, 1)
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=4.0, value=3.5, step=0.01)
        study_satisfaction = st.slider("Study Satisfaction (1-5)", 1, 5, 3)

    with col2:
        job_satisfaction = st.slider("Job Satisfaction (1-5)", 1, 5, 1)
        sleep_duration = st.selectbox("Sleep Duration", ["Less Than 5 Hours", "5-6 Hours", "7-8 Hours", "More Than 8 Hours"])
        dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
        degree = st.text_input("Degree", "S1")
        suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts ?", ["Yes", "No"])
        work_study_hours = st.number_input("Work/Study Hours", min_value=0, max_value=24, value=8)
        financial_stress = st.slider("Financial Stress (1-5)", 1, 5, 3)
        family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

    # Tombol dengan type="primary" agar CSS di atas bisa menargetkan tombol ini
    submitted = st.form_submit_button("Analisis Depresi", use_container_width=True, type="primary")

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

        prediction = model.predict(data_df)
        
        st.divider()
        if prediction[0] == 1:
            st.error("⚠️ Hasil: Berisiko Depresi")
        else:
            st.success("✅ Hasil: Tidak Berisiko")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")
        st.info("Saran: Cek kembali apakah teks pilihan sudah sama persis dengan dataset asli.")

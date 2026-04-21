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

    # 1. CSS untuk menyembunyikan tombol asli & mempercantik tombol merah
    st.markdown("""
        <style>
            div[data-testid="stForm"] button[kind="primaryFormSubmit"] {
                display: none !important;
            }
            .my-red-button {
                background-color: #FF4A4A;
                color: white;
                border-radius: 25px;
                border: none;
                padding: 15px;
                font-size: 18px;
                font-weight: bold;
                width: 100%;
                cursor: pointer;
                text-align: center;
                margin-bottom: 10px;
            }
            .my-red-button:hover {
                background-color: #E63939;
            }
        </style>
    """, unsafe_allow_html=True)

    # 2. Tombol Merah Kustom
    st.markdown('<button class="my-red-button" onclick="clickRealButton()">🔍 Prediksi Sekarang</button>', unsafe_allow_html=True)

    # 3. Javascript untuk memicu klik pada tombol asli
    st.components.v1.html("""
        <script>
            function clickRealButton() {
                var buttons = window.parent.document.querySelectorAll('button[kind="primaryFormSubmit"]');
                if (buttons.length > 0) {
                    buttons[buttons.length - 1].click();
                }
            }
            window.clickRealButton = clickRealButton;
        </script>
    """, height=0)

    # 4. Tombol asli (disembunyikan oleh CSS di atas)
    submitted = st.form_submit_button("Analisis Sekarang")

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
        
        # Bersihkan spasi pada data teks
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

import streamlit as st
import pandas as pd
import joblib

# Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Kesehatan Mental", layout="centered")

# Load Model (Hanya dilakukan sekali agar cepat)
@st.cache_resource
def load_data():
    return joblib.load('model_final_depresi.pkl')

model = load_data()

st.title("🧠 Deteksi Risiko Depresi Mahasiswa")
st.write("Aplikasi ini membantu memprediksi kecenderungan depresi berdasarkan data akademik dan pola hidup.")

# Form Input
with st.form("main_form"):
    st.subheader("Informasi Mahasiswa")
    
    # Kolom Kiri
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Umur", 17, 50, 20)
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        city = st.selectbox("Tipe Kota", ["Tier 1", "Tier 2"])
        academic_pressure = st.slider("Tekanan Akademik (1-5)", 1, 5, 3)
        cgpa = st.number_input("CGPA", 0.0, 10.0, 3.5)

    # Kolom Kanan
    with col2:
        sleep_duration = st.selectbox("Durasi Tidur", ["Less Than 5 Hours", "5-6 Hours", "7-8 Hours", "More Than 8 Hours"])
        dietary_habits = st.selectbox("Pola Makan", ["Healthy", "Moderate", "Unhealthy"])
        degree = st.selectbox("Jenjang", ["High School", "Bachelors", "Masters", "Doctorate"])
        study_satisfaction = st.slider("Kepuasan Studi (1-5)", 1, 5, 3)

    submitted = st.form_submit_button("Cek Sekarang")

if submitted:
    # Buat DataFrame dari input
    input_df = pd.DataFrame({
        'Age': [age], 'Gender': [gender], 'City': [city],
        'Academic Pressure': [academic_pressure], 'CGPA': [cgpa],
        'Sleep Duration': [sleep_duration], 'Dietary Habits': [dietary_habits],
        'Degree': [degree], 'Study Satisfaction': [study_satisfaction]
    })
    
    # Jalankan Prediksi
    res = model.predict(input_df)
    prob = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    if res[0] == 1:
        st.error(f"⚠️ **Hasil: Berisiko Depresi** (Probabilitas: {prob:.2%})")
        st.write("Tetap semangat! Jangan ragu untuk berdiskusi dengan orang yang kamu percayai.")
    else:
        st.success(f"✅ **Hasil: Tidak Berisiko** (Probabilitas: {1-prob:.2%})")
        st.write("Pertahankan pola hidup sehatmu!")

st.caption("Aplikasi ini dibuat untuk tujuan edukasi (Tugas Fast Track).")

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

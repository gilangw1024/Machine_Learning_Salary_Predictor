import streamlit as st
import pandas as pd
import joblib
from pipeline import preprocess_for_prediction

# Judul aplikasi
st.title("Salary Predictor")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('random_forest_best.pkl')

model = load_model()

# Daftar skill yang digunakan saat training
all_skills = [
    'airflow','aws','amazon','azure','bash','database','deep learning','docker','gcp','git','hadoop','java','keras',
    'kubernetes','linux','machine learning','matplotlib','neural network','numpy','sql','python','excel','tableau',
    'powerbi','opency','pandas','pytorch','r','scala','scikit-learn','scipy','sklearn','spark','tensorflow'
]

# Fungsi encoding skill manual
def encode_skills(skills, skill_pool):
    skill_list = [s.strip().lower() for s in skills.split(',')]
    return [1 if skill in skill_list else 0 for skill in skill_pool]

# Input pengguna
company_size = st.text_input("Ukuran perusahaan (contoh: 10-50, 1000, etc)")
skills = st.text_input("Skill (pisahkan dengan koma)")
status = st.selectbox("Status kerja", ["remote", "hybrid", "onsite", "unknown"])
industry = st.selectbox("Industri", ["technology", "manufacturing", "retail", "education", "energy", "finance", "healthcare", "logistics"])
ownership = st.selectbox("Kepemilikan", ["public", "private", "unknown"])
job = st.selectbox("Jenis pekerjaan", ["data analyst", "data scientist", "machine learning", "data engineer"])
seniority_level = st.selectbox("Level senioritas", ["junior", "midlevel", "senior", "lead"])

# Tombol prediksi
if st.button("Prediksi Gaji"):
    if not skills or not company_size:
        st.warning("Masukkan ukuran perusahaan dan minimal satu skill.")
        st.stop()

    # Bentuk DataFrame input
    df_input = pd.DataFrame([{
        'company_size': company_size,
        'skills': str([s.strip().lower() for s in skills.split(',')]),  # bentuk list string
        'status': status,
        'industry': industry,
        'ownership': ownership,
        'job_title': job,
        'seniority_level': seniority_level,
        **dict(zip([f"skill_{s}" for s in all_skills], encode_skills(skills, all_skills)))
    }])

    # Preprocessing
    df_ready = preprocess_for_prediction(df_input)

    # Prediksi
    try:
        prediction = model.predict(df_ready)
        st.success(f"Perkiraan gaji: Rp {int(prediction[0]):,}")
    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")

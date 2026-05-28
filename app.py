import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="ARCENE Kanser Sınıflandırma", layout="centered")

st.title("ARCENE Kanser Sınıflandırma Uygulaması")

st.write("""
Bu uygulama, ARCENE veri seti kullanılarak eğitilmiş makine öğrenmesi modeliyle
örneğin kanserli veya normal olduğunu tahmin eder.
""")

model = joblib.load("arcene_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

st.subheader("Veri Girişi")

st.info("ARCENE veri setinde her örnek 10.000 öznitelikten oluşur. Bu nedenle manuel giriş yerine CSV dosyası yükleyiniz.")

uploaded_file = st.file_uploader("10.000 öznitelikli CSV dosyası yükleyiniz", type=["csv"])

if uploaded_file is not None:
    data = np.loadtxt(uploaded_file, delimiter=",")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    st.write("Yüklenen veri boyutu:", data.shape)

    if data.shape[1] != 10000:
        st.error("Hata: Veri 10.000 öznitelik içermelidir.")
    else:
        data_scaled = scaler.transform(data)
        data_pca = pca.transform(data_scaled)

        prediction = model.predict(data_pca)

        st.subheader("Tahmin Sonucu")

        for i, pred in enumerate(prediction):
            if pred == 1:
                st.error(f"{i+1}. örnek tahmini: Kanserli")
            else:
                st.success(f"{i+1}. örnek tahmini: Normal")
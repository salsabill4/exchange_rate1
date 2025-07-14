import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Setup halaman
st.set_page_config(page_title="Prediksi Nilai Tukar", layout="wide")
st.title("📈 Prediksi Nilai Tukar terhadap Dolar (Prophet + LSTM)")

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload file CSV nilai tukar (harus berisi kolom 'date' dan 'price')", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    # Gunakan file default
    data = pd.read_csv("exchange_rate0.csv")

# Preprocessing awal
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(by='date', ascending=False)
data = data[['date', 'price']].rename(columns={'date': 'ds', 'price': 'y'})

# Sidebar: pilihan durasi prediksi
st.sidebar.header("Pengaturan Prediksi")
periode_dict = {
    "1 Bulan": 30,
    "3 Bulan": 90,
    "6 Bulan": 180,
    "9 Bulan": 270,
    "1 Tahun": 365
}
periode_label = st.sidebar.selectbox("Pilih rentang waktu prediksi:", list(periode_dict.keys()))
periode_hari = periode_dict[periode_label]

# ===================== 1. Train Prophet ===================== #
with st.spinner("⏳ Melatih model Prophet..."):
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(data)
    future = prophet_model.make_future_dataframe(periods=periode_hari)
    forecast = prophet_model.predict(future)
    df_forecast = forecast[['ds', 'yhat']]
    df_merged = pd.merge(data, df_forecast, on='ds', how='inner')
    df_merged['residual'] = df_merged['y'] - df_merged['yhat']

# ===================== 2. LSTM on Residuals ===================== #
residual = df_merged['residual'].dropna().values.reshape(-1, 1)
scaler = MinMaxScaler()
residual_scaled = scaler.fit_transform(residual)

def create_sequences(datax, seq_len):
    X, y = [], []
    for i in range(len(datax) - seq_len):
        X.append(datax[i:i+seq_len])
        y.append(datax[i+seq_len])
    return np.array(X), np.array(y)

SEQ_LEN = 8
X, y_seq = create_sequences(residual_scaled, SEQ_LEN)

with st.spinner("⏳ Melatih model LSTM untuk residual..."):
    model = Sequential([
        LSTM(50, input_shape=(SEQ_LEN, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y_seq, epochs=70, batch_size=8, verbose=0)

# ===================== 3. Prediksi ke Depan ===================== #
last_sequence = residual_scaled[-SEQ_LEN:]
predicted_future_residual = []

for _ in range(periode_hari):
    input_seq = last_sequence.reshape(1, SEQ_LEN, 1)
    pred_scaled = model.predict(input_seq, verbose=0)
    predicted_future_residual.append(pred_scaled[0, 0])
    last_sequence = np.append(last_sequence[1:], pred_scaled, axis=0)

# Invers transform
predicted_future_residual = scaler.inverse_transform(np.array(predicted_future_residual).reshape(-1, 1)).flatten()

# Ambil yhat Prophet dan tambah residual
future_only = forecast.tail(periode_hari)
final_forecast = future_only['yhat'].values + predicted_future_residual

# ===================== 4. Tampilkan Output ===================== #
st.subheader(f"📆 Hasil Prediksi {periode_label}")
selected_date = st.date_input("Pilih tanggal yang ingin dilihat prediksinya:", value=future_only['ds'].iloc[0].date(), min_value=future_only['ds'].iloc[0].date(), max_value=future_only['ds'].iloc[-1].date())

selected_pred = final_forecast[list(future_only['ds'].dt.date).index(selected_date)]
st.metric(label="📌 Prediksi nilai tukar pada tanggal tersebut", value=f"${selected_pred:.2f}")

# Tampilkan tabel hasil prediksi
result_df = pd.DataFrame({
    'Tanggal': future_only['ds'],
    'Prediksi Harga': final_forecast
})
st.dataframe(result_df)

# ===================== 5. Visualisasi ===================== #
st.subheader("📊 Grafik Prediksi Nilai Tukar")
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(data['ds'], data['y'], label='Harga Historis')
ax.plot(future_only['ds'], final_forecast, label='Prediksi Harga', color='red')
ax.axvline(data['ds'].max(), color='gray', linestyle='--', label='Hari Ini')
ax.set_title('Prediksi Nilai Tukar terhadap Dolar')
ax.set_xlabel('Tanggal')
ax.set_ylabel('Harga')
ax.legend()
ax.grid(True)
st.pyplot(fig)


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI Dự đoán Tác nhân & Gợi ý Kháng sinh")
st.title("🧬 AI Dự đoán Tác nhân và Gợi ý Kháng sinh")

@st.cache_data
def train_model():
    df = pd.read_csv("Mô hình AI.csv")
    df = df[df["Tac nhan"] != "unspecified"]  # Loại nhãn

    X = df.drop(columns=["Tac nhan"])
    y = df["Tac nhan"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y_encoded)
    return model, X.columns.tolist(), label_encoder

model, feature_cols, label_encoder = train_model()

# Nhập dữ liệu
st.header("📋 Nhập dữ liệu lâm sàng")
user_input = {}

def binary_input(label):
    return 1 if st.radio(label, ["Không", "Có"], horizontal=True) == "Có" else 0

for col in feature_cols:
    if col.lower() in ["sot", "ho", "chay mui", "dam", "non oi", "tieu chay", "kich thich quay khoc",
                      "tho met", "tho nhanh", "bo an kem", "dong dac phoi phai", "dong dac phoi trai",
                      "ran phoi", "su dung khang sinh truoc khi den vien"]:
        user_input[col] = binary_input(col)
    else:
        user_input[col] = st.number_input(col, value=0.0, format="%.2f")

if st.button("🔍 Dự đoán"):
    df_input = pd.DataFrame([user_input])
    prediction = model.predict(df_input[feature_cols])[0]
    label = label_encoder.inverse_transform([prediction])[0]
    st.success(f"✅ Tác nhân gây bệnh được dự đoán: **{label}**")

    st.subheader("💊 Kháng sinh gợi ý:")
    if label == "RSV":
        st.info("RSV là virus, không gợi ý kháng sinh")
    elif label == "M. pneumonia":
        st.write("**Thường dùng:** Macrolide (Azithromycin, Clarithromycin), Tetracycline")
    else:
        try:
            df_k = pd.read_csv("Mô hình KSD.csv")
            row = df_k[df_k["Tac nhan"] == label].drop(columns=["Tac nhan"])
            if not row.empty:
                antibiotics = row.columns[row.values[0] == 0.5].tolist()
                if antibiotics:
                    st.write("Kháng sinh nhạy (gợi ý):", ", ".join(antibiotics))
                else:
                    st.write("Không có kháng sinh nào được gợi ý.")
            else:
                st.warning("Không tìm thấy dữ liệu kháng sinh.")
        except Exception as e:
            st.error("Lỗi khi đọc file kháng sinh gợi ý.")

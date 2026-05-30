import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array

# =========================
# CẤU HÌNH TRANG
# =========================
st.set_page_config(
    page_title="AD Money Recognition",
    page_icon="💵",
    layout="centered"
)

# =========================
# CSS GIAO DIỆN
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e8f5e9, #ffffff);
}

.main-title {
    text-align: center;
    color: #1b5e20;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    color: #4e4e4e;
    font-size: 18px;
    margin-bottom: 30px;
}

.result-box {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.12);
    text-align: center;
    margin-top: 20px;
}

.money-result {
    font-size: 36px;
    font-weight: bold;
    color: #1b5e20;
}

.confidence {
    font-size: 22px;
    color: #333333;
}

.info-box {
    background-color: #f1f8e9;
    padding: 18px;
    border-radius: 15px;
    border-left: 6px solid #43a047;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TIÊU ĐỀ
# =========================
st.markdown('<div class="main-title">💵 AD Money Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ứng dụng AI nhận diện mệnh giá tiền Việt Nam</div>', unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_money_model():
    model = load_model("vnd_money_model.h5")
    return model

model = load_money_model()

# =========================
# LABEL
# =========================
class_labels = {
    0: "10000",
    1: "100000",
    2: "20000",
    3: "200000",
    4: "50000",
    5: "500000"
}

# =========================
# HÀM DỰ ĐOÁN
# =========================
def predict_money(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    img_array = img_to_array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    predicted_label = class_labels[predicted_index]

    return predicted_label, confidence

# =========================
# CHỌN CHỨC NĂNG
# =========================
st.markdown("### Chọn cách nhập ảnh")

option = st.radio(
    "Anh muốn dùng cách nào?",
    ["Tải ảnh lên", "Chụp ảnh trực tiếp"],
    horizontal=True
)

image = None

if option == "Tải ảnh lên":
    uploaded_file = st.file_uploader(
        "Tải ảnh tờ tiền lên",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

elif option == "Chụp ảnh trực tiếp":
    camera_file = st.camera_input("Chụp ảnh tờ tiền")

    if camera_file is not None:
        image = Image.open(camera_file)

# =========================
# HIỂN THỊ ẢNH VÀ DỰ ĐOÁN
# =========================
if image is not None:
    st.image(image, caption="Ảnh đã chọn", use_container_width=True)

    if st.button("🔍 Nhận diện mệnh giá"):
        label, confidence = predict_money(image)

        st.markdown(f"""
        <div class="result-box">
            <div class="money-result">{label} VND</div>
            <div class="confidence">Độ tin cậy: {confidence:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        if confidence < 60:
            st.warning("Độ tin cậy hơi thấp. Anh nên chụp rõ tờ tiền hơn, đủ sáng và không bị che khuất.")
        else:
            st.success("Nhận diện thành công!")

else:
    st.markdown("""
    <div class="info-box">
        <b>Hướng dẫn:</b><br>
        1. Tải ảnh tiền Việt lên hoặc chụp trực tiếp bằng camera.<br>
        2. Nhấn nút <b>Nhận diện mệnh giá</b>.<br>
        3. App sẽ hiển thị mệnh giá dự đoán và độ tin cậy.
    </div>
    """, unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Developed by AD</p>",
    unsafe_allow_html=True
)
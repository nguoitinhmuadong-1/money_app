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
/* Nền chính */
.stApp {
    background: linear-gradient(135deg, #0f2f1f, #1b4332, #2d6a4f);
}

/* Tiêu đề chính */
.main-title {
    text-align: center;
    color: #ffffff !important;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 5px;
}

/* Tiêu đề phụ */
.sub-title {
    text-align: center;
    color: #d8f3dc !important;
    font-size: 18px;
    margin-bottom: 30px;
}

/* Chữ tổng quát trên nền tối */
h1, h2, h3, h4, h5, h6, p, label, span {
    color: #ffffff !important;
}

/* Chữ markdown */
[data-testid="stMarkdownContainer"] {
    color: #ffffff !important;
}

/* Radio */
.stRadio label {
    color: #ffffff !important;
    font-weight: 500 !important;
}

.stRadio div {
    color: #ffffff !important;
}

/* File uploader label */
.stFileUploader label {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Camera label */
.stCameraInput label {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Khung upload */
[data-testid="stFileUploaderDropzone"] {
    background-color: #ffffff !important;
    border: 2px dashed #95d5b2 !important;
    border-radius: 15px !important;
}

/* Chữ trong khung upload */
[data-testid="stFileUploaderDropzone"] div {
    color: #1f2937 !important;
}

/* File đã upload */
[data-testid="stFileUploaderFile"] {
    background-color: #e9f5ec !important;
    color: #1f2937 !important;
    border-radius: 12px !important;
}

[data-testid="stFileUploaderFile"] div {
    color: #1f2937 !important;
}

/* Nút bấm */
.stButton button {
    background-color: #52b788 !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
    font-weight: bold !important;
    border: none !important;
    width: 100%;
}

.stButton button:hover {
    background-color: #40916c !important;
    color: white !important;
}

/* Ảnh hiển thị */
[data-testid="stImage"] {
    border-radius: 18px !important;
}

/* Khung kết quả */
.result-box {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.25);
    text-align: center;
    margin-top: 20px;
    border: 2px solid #95d5b2;
}

.money-result {
    font-size: 38px;
    font-weight: bold;
    color: #1b5e20 !important;
}

.confidence {
    font-size: 22px;
    color: #333333 !important;
    margin-top: 8px;
}

/* Khung hướng dẫn */
.info-box {
    background-color: #ffffff;
    color: #1f2937 !important;
    padding: 18px;
    border-radius: 15px;
    border-left: 6px solid #52b788;
    margin-top: 20px;
    line-height: 1.7;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.18);
}

.info-box b {
    color: #1b5e20 !important;
}

.info-box div, .info-box span, .info-box p {
    color: #1f2937 !important;
}

/* Success / warning text dễ đọc */
.stAlert div {
    color: #1f2937 !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #d8f3dc !important;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TIÊU ĐỀ
# =========================
st.markdown(
    '<div class="main-title">💵 AD Money Recognition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Ứng dụng AI nhận diện mệnh giá tiền Việt Nam</div>',
    unsafe_allow_html=True
)

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
# Giữ nguyên label KHÔNG có VND
# Phải đúng với train_generator.class_indices lúc train
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
# GIAO DIỆN NHẬP ẢNH
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
# HIỂN THỊ ẢNH VÀ KẾT QUẢ
# =========================
if image is not None:
    st.image(
        image,
        caption="Ảnh đã chọn",
        use_container_width=True
    )

    if st.button("🔍 Nhận diện mệnh giá"):
        label, confidence = predict_money(image)

        st.markdown(f"""
        <div class="result-box">
            <div class="money-result">{label} VND</div>
            <div class="confidence">Độ tin cậy: {confidence:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        if confidence < 60:
            st.warning(
                "Độ tin cậy hơi thấp. Anh nên chụp ảnh rõ hơn, đủ sáng, không bị che khuất và thấy rõ toàn bộ tờ tiền."
            )
        else:
            st.success("Nhận diện thành công!")

else:
    st.markdown("""
    <div class="info-box">
        <b>Hướng dẫn sử dụng:</b><br>
        1. Chọn <b>Tải ảnh lên</b> hoặc <b>Chụp ảnh trực tiếp</b>.<br>
        2. Đưa ảnh tờ tiền Việt Nam vào hệ thống.<br>
        3. Nhấn nút <b>Nhận diện mệnh giá</b>.<br>
        4. App sẽ hiển thị mệnh giá dự đoán và độ tin cậy.
    </div>
    """, unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    '<p class="footer">Developed by AD</p>',
    unsafe_allow_html=True
)
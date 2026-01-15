import os
import streamlit as st
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import gdown

# =============================
# CONFIG
# =============================
DEVICE = torch.device("cpu")

MODEL_PATH = "oil_spill_unet.pth"
MODEL_ID = "1JWz4xx7sVja-dZFB7lIYK8aGvFNRF7mA"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Oil Spill Detection",
    page_icon="🛢️",
    layout="wide"
)

# =============================
# CUSTOM CSS (FRONTEND)
# =============================
st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
    }
    .title {
        font-size: 42px;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 18px;
        color: #B0B3B8;
        margin-bottom: 30px;
    }
    .card {
        background-color: #161B22;
        padding: 20px;
        border-radius: 14px;
        margin-bottom: 20px;
    }
    .verdict-green {
        background-color: #0F5132;
        color: #75F0B5;
        padding: 12px 18px;
        border-radius: 30px;
        font-size: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .verdict-red {
        background-color: #5A1111;
        color: #FF7B7B;
        padding: 12px 18px;
        border-radius: 30px;
        font-size: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .metric-label {
        color: #B0B3B8;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================
# DOWNLOAD MODEL
# =============================
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return MODEL_PATH

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

download_model()
model = load_model()

# =============================
# HEADER
# =============================
st.markdown('<div class="title">🛢️ Oil Spill Detection System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-based satellite image analysis for detecting marine oil spills</div>',
    unsafe_allow_html=True
)

# =============================
# UPLOAD SECTION
# =============================
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "📤 Upload Satellite Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)
st.markdown('</div>', unsafe_allow_html=True)

# =============================
# PROCESS IMAGES
# =============================
if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(f"📄 {uploaded_file.name}")

        # -------- Read Image --------
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (256, 256))

        # -------- Preprocess --------
        img = image_resized.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)

        # -------- Prediction --------
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits)

        mask = (probs > 0.5).cpu().numpy().astype(np.uint8).squeeze()
        prob_map = probs.squeeze().cpu().numpy()

        spill_pixels = np.sum(mask == 1)
        total_pixels = mask.size
        spill_percentage = (spill_pixels / total_pixels) * 100

        if spill_pixels > 0:
            confidence = prob_map[mask == 1].mean() * 100
            verdict_html = '<div class="verdict-red">🛢️ Oil Spill Detected</div>'
        else:
            confidence = (1 - prob_map).mean() * 100
            verdict_html = '<div class="verdict-green">✅ No Oil Spill Detected</div>'

        # -------- Images --------
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_resized, caption="Input Image", width="stretch")
        with col2:
            overlay = image_resized.copy()
            overlay[mask == 1] = [255, 0, 0]
            result = cv2.addWeighted(image_resized, 0.7, overlay, 0.3, 0)
            st.image(result, caption="Prediction Overlay", width="stretch")

        st.markdown(verdict_html, unsafe_allow_html=True)

        # -------- Metrics --------
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Prediction Confidence (%)", f"{confidence:.2f}")
        with m2:
            st.metric("Spill Area (%)", f"{spill_percentage:.3f}")

        st.progress(min(int(confidence), 100))

        st.markdown('</div>', unsafe_allow_html=True)

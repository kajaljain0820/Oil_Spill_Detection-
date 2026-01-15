import os
import streamlit as st
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import gdown
from PIL import Image

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
    page_title="Oil Spill Detection System",
    page_icon="🛢️",
    layout="wide"
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
# UI
# =============================
st.title("🛢️ Oil Spill Detection System")
st.markdown(
    """
    Upload **satellite images** to detect oil spills.
    - Pixel-accurate segmentation  
    - Area-based detection (correct logic)  
    - Region-based confidence  
    - Batch processing  
    """
)

st.divider()

uploaded_files = st.file_uploader(
    "📤 Upload Satellite Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# =============================
# PROCESS IMAGES
# =============================
if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"📄 {uploaded_file.name}")

        # -------- Read Image --------
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (256, 256))

        # -------- Preprocess (MATCH COLAB) --------
        img = image_resized.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)

        # -------- Prediction --------
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits)

        mask = (probs > 0.5).cpu().numpy().astype(np.uint8).squeeze()

        # -------- Correct Detection Logic --------
        spill_pixels = np.sum(mask == 1)
        total_pixels = mask.size
        spill_percentage = (spill_pixels / total_pixels) * 100

        if spill_pixels > 0:
            confidence = probs.cpu().numpy()[mask == 1].mean() * 100
            verdict = "🛢️ Oil Spill Detected"
            verdict_color = "red"
        else:
            confidence = 0.0
            verdict = "✅ No Oil Spill Detected"
            verdict_color = "green"

        # -------- Overlay --------
        overlay = image_resized.copy()
        overlay[mask == 1] = [255, 0, 0]
        result = cv2.addWeighted(image_resized, 0.7, overlay, 0.3, 0)

        # -------- Display --------
        col1, col2 = st.columns(2)

        with col1:
            st.image(image_resized, caption="Input Image", use_container_width=True)

        with col2:
            st.image(result, caption="Predicted Oil Spill", use_container_width=True)

        st.markdown(
            f"<h3 style='color:{verdict_color}'>{verdict}</h3>",
            unsafe_allow_html=True
        )

        # -------- Metrics --------
        st.metric("Model Confidence (%)", f"{confidence:.2f}")
        st.metric("Spill Area (%)", f"{spill_percentage:.3f}")

        # -------- Progress Bar --------
        st.write("### 🔍 Confidence Level")
        st.progress(min(int(confidence), 100))

        # -------- Download --------
        st.download_button(
            label="⬇️ Download Result Image",
            data=cv2.imencode(".png", result)[1].tobytes(),
            file_name=f"result_{uploaded_file.name}",
            mime="image/png"
        )

        st.divider()

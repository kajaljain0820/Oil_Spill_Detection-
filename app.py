import os
import streamlit as st
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import gdown
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = torch.device("cpu")

MODEL_PATH = "oil_spill_unet.pth"
MODEL_ID = "1JWz4xx7sVja-dZFB7lIYK8aGvFNRF7mA"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Oil Spill Detection System",
    page_icon="🛢️",
    layout="wide"
)

# -----------------------------
# DOWNLOAD MODEL
# -----------------------------
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return MODEL_PATH

# -----------------------------
# LOAD MODEL
# -----------------------------
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

# -----------------------------
# UI
# -----------------------------
st.title("🛢️ Oil Spill Detection System")
st.markdown(
    """
    **Features**
    - Upload **multiple satellite images**
    - Get **oil spill segmentation**
    - See **confidence progress bar**
    - **Download result image**
    """
)

st.divider()

uploaded_files = st.file_uploader(
    "📤 Upload Satellite Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# -----------------------------
# PROCESS IMAGES
# -----------------------------
if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"📄 {uploaded_file.name}")

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (256, 256))

        img_tensor = torch.tensor(image_resized).permute(2, 0, 1)
        img_tensor = img_tensor.unsqueeze(0).float() / 255.0

        with torch.no_grad():
            output = model(img_tensor)
            output = torch.sigmoid(output)

        prob_map = output.squeeze().cpu().numpy()
        mask = (prob_map > 0.5).astype(np.uint8)

        # -----------------------------
        # CONFIDENCE
        # -----------------------------
        confidence = float(prob_map.mean()) * 100

        if confidence >= 20:
            verdict = "🛢️ Oil Spill Detected"
            verdict_color = "red"
        else:
            verdict = "✅ No Oil Spill Detected"
            verdict_color = "green"

        # -----------------------------
        # CREATE OVERLAY IMAGE
        # -----------------------------
        overlay = image_resized.copy()
        overlay[mask == 1] = [255, 0, 0]  # red overlay
        result = cv2.addWeighted(image_resized, 0.7, overlay, 0.3, 0)

        # Convert for download
        result_pil = Image.fromarray(result)
        img_buffer = np.array(result_pil)

        # -----------------------------
        # DISPLAY
        # -----------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.image(image_resized, caption="Input Image", use_container_width=True)

        with col2:
            st.image(result, caption="Result (Overlay)", use_container_width=True)

        st.markdown(
            f"<h3 style='color:{verdict_color}'>{verdict}</h3>",
            unsafe_allow_html=True
        )

        # -----------------------------
        # PROGRESS BAR CONFIDENCE
        # -----------------------------
        st.write("### 🔍 Confidence Level")
        st.progress(int(confidence))
        st.write(f"**{confidence:.2f}%**")

        # -----------------------------
        # DOWNLOAD BUTTON
        # -----------------------------
        st.download_button(
            label="⬇️ Download Result Image",
            data=cv2.imencode(".png", result)[1].tobytes(),
            file_name=f"result_{uploaded_file.name}",
            mime="image/png"
        )

        st.divider()

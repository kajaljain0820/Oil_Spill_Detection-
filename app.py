import os
import streamlit as st
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import gdown

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = torch.device("cpu")  # Streamlit Cloud is CPU-only

MODEL_PATH = "oil_spill_unet.pth"
MODEL_ID = "1JWz4xx7sVja-dZFB7lIYK8aGvFNRF7mA"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"

# -----------------------------
# DOWNLOAD MODEL (ONLY ONCE)
# -----------------------------
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return MODEL_PATH

# -----------------------------
# LOAD MODEL (CACHED)
# -----------------------------
@st.cache_resource
def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    )
    model.to(DEVICE)
    model.eval()
    return model


# -----------------------------
# APP UI
# -----------------------------
st.set_page_config(
    page_title="Oil Spill Detection",
    layout="centered"
)

st.title("🛢️ Oil Spill Detection System")
st.write("Upload a satellite image to detect oil spill regions.")

# Ensure model is ready
download_model()
model = load_model()

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a satellite image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize
    image_resized = cv2.resize(image, (256, 256))

    # Prepare tensor
    img_tensor = torch.tensor(image_resized).permute(2, 0, 1)
    img_tensor = img_tensor.unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(DEVICE)

    # Prediction
    with torch.no_grad():
        output = model(img_tensor)
        output = torch.sigmoid(output)
        mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    # -----------------------------
    # DISPLAY RESULTS
    # -----------------------------
    st.subheader("📷 Input Image")
    st.image(image_resized, use_container_width=True)

    st.subheader("🧠 Predicted Oil Spill Mask")
    st.image(mask * 255, clamp=True, use_container_width=True)

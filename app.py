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
    layout="centered"
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
# FRONTEND UI
# -----------------------------
st.title("🛢️ Oil Spill Detection System")
st.markdown(
    """
    Upload a **satellite image** to detect oil spill regions.
    The system will provide:
    - 🧠 Oil spill segmentation
    - 📊 Confidence score
    - ✅ Final decision
    """
)

st.divider()

uploaded_file = st.file_uploader(
    "📤 Upload Satellite Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# PROCESS IMAGE
# -----------------------------
if uploaded_file is not None:
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
    # CONFIDENCE CALCULATION
    # -----------------------------
    confidence = float(prob_map.mean()) * 100

    # Decision threshold
    if confidence >= 20:
        verdict = "🛢️ Oil Spill Detected"
        verdict_color = "red"
    else:
        verdict = "✅ No Oil Spill Detected"
        verdict_color = "green"

    # -----------------------------
    # DISPLAY RESULTS
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Input Image")
        st.image(image_resized, use_container_width=True)

    with col2:
        st.subheader("🧠 Predicted Mask")
        st.image(mask * 255, clamp=True, use_container_width=True)

    st.divider()

    # Verdict & Confidence
    st.markdown(
        f"""
        <h2 style="color:{verdict_color}; text-align:center;">
            {verdict}
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.metric(
        label="Model Confidence",
        value=f"{confidence:.2f} %"
    )


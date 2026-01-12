import os
import streamlit as st
import gdown
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model download from Google Drive
MODEL_PATH = "oil_spill_unet.pth"
MODEL_ID = "1JWz4xx7sVja-dZFB7lIYK8aGvFNRF7mA"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model, please wait...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Define the model
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

st.title("Oil Spill Detection System")

# Upload image
uploaded_file = st.file_uploader(
    "Upload a satellite image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize
    image = cv2.resize(image, (256, 256))

    # Prepare tensor
    img_tensor = torch.tensor(image).permute(2, 0, 1)
    img_tensor = img_tensor.unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(DEVICE)

    # Prediction
    with torch.no_grad():
        output = model(img_tensor)
        output = torch.sigmoid(output)
        mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    # Show results
    st.subheader("Input Image")
    st.image(image)

    st.subheader("Predicted Oil Spill Mask")
    st.image(mask * 255, clamp=True)

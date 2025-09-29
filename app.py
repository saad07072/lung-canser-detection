# app.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

MODEL_PATH = "mini_model.h5"
IMG_SIZE = (128,128)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

st.title("Mini Lung Cancer Demo")
st.write("Partial demo (synthetic dataset). Replace with your CT images for a real demo.")

model = load_model()

uploaded = st.file_uploader("Upload a chest image (PNG/JPG)", type=["png","jpg","jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("L")
    st.image(img, caption="Uploaded (grayscale)", use_column_width=True)
    # preprocess
    img_resized = img.resize(IMG_SIZE)
    arr = np.array(img_resized).astype("float32")/255.0
    arr = arr.reshape((1, IMG_SIZE[0], IMG_SIZE[1], 1))
    pred = model.predict(arr)[0][0]
    st.metric("Cancer probability (0..1)", float(pred))
    st.write("Interpretation:")
    if pred >= 0.5:
        st.error("Model predicts: **CANCER** (demo)")
    else:
        st.success("Model predicts: **NORMAL** (demo)")
else:
    st.info("Upload an image to get a prediction.")

st.write("Model file:", MODEL_PATH)

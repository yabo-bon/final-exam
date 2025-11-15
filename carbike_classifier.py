import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="Car/Bike Classifier", layout="centered")

# ----------------- CUSTOM STYLING -----------------
st.markdown("""
    <style>
    .stApp {             
        background-color: #1c1c1c !important;
        color: #ffffff;
    }
    h1, h2, h3, p, label {
        color: #FFD700 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.4);
    }
    h1 {
        text-align: center;
    }
    .instructions-box {
        background-color: rgba(255, 215, 0, 0.15);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #FFD700;
        margin-bottom: 25px;
    }
    .stButton>button {
        background-color: #FFD700 !important;
        color: #000 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: bold !important;
        border: none;
    }
    .stFileUploader label {
        color: #FFD700 !important;
        font-weight: bold !important;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- LOAD MODEL -----------------
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('car_bike_model.keras')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure your trained model is saved as 'car_bike_model.keras' in the same directory.")
        return None

model = load_model()
class_names = ['Bike', 'Car']

# ----------------- TITLE -----------------
st.title("🚗🏍️ Car or Bike Image Classifier")

# ----------------- HOW TO USE -----------------
st.markdown("""
<div class="instructions-box">
    <h3>📌 How to Use This App</h3>
    <p>1. Click the <strong>Upload</strong> button below and choose a picture of a car or motorcycle/bike.</p>
    <p>2. Wait a few seconds while the model analyzes the image.</p>
    <p>3. The classifier will display whether the image is a <strong>Car</strong> or <strong>Motorcycle/Bike</strong>, along with its confidence level.</p>
    <p>4. Try different angles, lighting conditions, and vehicle types for best results.</p>
</div>
""", unsafe_allow_html=True)

file = st.file_uploader("Upload a clear image of a vehicle", type=["jpg", "jpeg", "png", "bmp"])

def import_and_predict(image_data, model):
    if model is None:
        return None

    size = model.input_shape[1:3]
    image = ImageOps.fit(image_data, size, Image.LANCZOS)

    img_array = np.asarray(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    return prediction

if file is None:
    st.info("Please upload an image of a car or bike to proceed.")
elif model is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Analyzing image..."):
        prediction = import_and_predict(image, model)

    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]
    confidence = round(100 * np.max(prediction), 2)

    st.markdown(f"### 🏁 Predicted Vehicle Type: **{predicted_class}**")
    st.markdown(f"**Confidence:** `{confidence}%`")

    if predicted_class == 'Car':
        st.success("🚗 This image appears to show a **Car**.")
    elif predicted_class == 'Bike':
        st.success("🏍️ This image appears to show a **Bike/Motorcycle**.")
    else:
        st.info("The model detected something, but it's unsure.")

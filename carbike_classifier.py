import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="Car/Bike Classifier", layout="centered")

st.markdown("""
    <style>
    h1, h2, h3, p, label, span {
        color: #ffffff !important;        
        text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
    }

    h1 {
        text-align: center;
        font-weight: 900;
    }

    .instructions-box {
        background-color: rgba(0, 0, 0, 0.65);
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #ffeb3b;
        margin-bottom: 25px;
    }

    .stButton>button {
        background-color: #ffeb3b !important;
        color: #000000 !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.3rem !important;
        border: none;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }

    .stFileUploader label {
        color: #ffeb3b !important;
        font-size: 17px;
        font-weight: bold;
    }

    </style>
""", unsafe_allow_html=True)

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

st.title("🚗🏍️ Car or Bike Image Classifier")

st.markdown("""
<div class="instructions-box">
    <h3>📌 How to Use This App</h3>
    <p>1. Click the <strong>Browse</strong> button below and choose a picture of a car or motorcycle/bike.</p>
    <p>2. Wait a few seconds while the model analyzes the image.</p>
    <p>3. The classifier will display whether the image is a <strong>Car</strong> or <strong>Motorcycle/Bike</strong>, along with its confidence level.</p>
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

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="Car/Bike Classifier", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background-color: #1b1b1b;
        color: #fdd835;
    }
    h1, h2, h3, p, label {
        color: #fdd835 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    h1 {
        text-align: center;
        font-size: 42px;
        color: #FFEB3B;
    }
    .uploadedImage > img {
        border: 3px solid #fdd835;
        border-radius: 10px;
        box-shadow: 0px 0px 15px rgba(255, 230, 0, 0.4);
    }
    .css-1v0mbdj, .css-ffhzg2 {
        color: #fdd835 !important;
    }
    .stButton>button {
        background-color: #fdd835;
        color: black;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        border: 2px solid black;
    }
    .stButton>button:hover {
        background-color: black;
        color: #fdd835;
        border: 2px solid #fdd835;
    }
    .stFileUploader label {
        color: #fdd835 !important;
        font-size: 18px;
    }
    .result-box {
        background-color: rgba(253, 216, 53, 0.1);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #fdd835;
        text-align: center;
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

st.title("Bike or Car Classifier")
st.write("Upload an image to determine if it is a bike/motorcycle or a car.")

file = st.file_uploader("Upload a vehicle image", type=["jpg", "jpeg", "png", "bmp"])

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

    with st.spinner("Analyzing image..."):
        prediction = import_and_predict(image, model)

    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]
    confidence = round(100 * np.max(prediction), 2)

    st.markdown(f"### 🏁 Predicted Vehicle Type: **{predicted_class}**")
    st.markdown(f"Confidence: `{confidence}%`")

    if predicted_class == 'Car':
        st.success("Classification successful! This looks like a Car.")
    elif predicted_class == 'Bike':
        st.success("Classification successful! This looks like a Bike/Motorcycle.")
    else:
        st.info("The model detected something, but it's inconclusive.")

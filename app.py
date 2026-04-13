import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Acne Detection AI", layout="centered")
st.title("AI Acne Detection App")
st.write("Upload a face image to analyze acne severity")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("acne_severity_model.keras")
    return model

model = load_model()

def preprocess_image(image):
    image = image.resize((128, 128))
    image = image.convert("L")
    image = np.array(image) / 255.0
    image = image.flatten()
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)

    return image  # ✅ was returning class 'Image' instead of the variable

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze"):
        with st.spinner("Analyzing..."):  # ✅ fixed broken 'with' block and typo
            processed = preprocess_image(image)  # ✅ fixed indentation
            prediction = model.predict(processed)

            # ✅ These were missing — added result/confidence extraction
            class_labels = ["Clear Skin", "Mild Acne", "Moderate Acne", "Severe Acne"]
            result = class_labels[np.argmax(prediction)]
            confidence = np.max(prediction)

            st.success(f"Result: {result}")
            st.write(f"Confidence: {confidence:.2f}")

            if result == "Clear Skin":
                st.info("hello")
            elif result == "Mild Acne":
                st.warning("You have a few whiteheads, blackheads, or small pimples.")
            elif result == "Moderate Acne":
                st.warning("You have multiple inflamed pimples, papules, or pustules.")
            else:
                st.error("Your skin shows signs of severe inflammatory acne.")

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image

# Load the saved model
loaded_model = load_model("food101_model.h5")

# Load Food101 dataset info to get class names
data_info = tfds.builder('food101').info
class_names = data_info.features['label'].names

# Streamlit interface
st.title("Food101 Classification")

# Upload image through Streamlit interface
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image
    image = image.resize((224, 224))
    image = tf.keras.applications.densenet.preprocess_input(np.array(image))
    image = np.expand_dims(image, axis=0)

    # Make a prediction
    prediction = loaded_model.predict(image)
    predicted_class = np.argmax(prediction)

    st.write("Prediction:", class_names[predicted_class])
    st.write("Confidence:", prediction[0][predicted_class])

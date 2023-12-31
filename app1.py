import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import PIL.Image as Image
import tensorflow_hub as hub

# Function to load the model with a custom object scope
def load_model_with_custom_objects(model_path):
    custom_objects = {"KerasLayer": hub.KerasLayer}
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# Load the pre-trained ResNet model with custom objects
model = load_model_with_custom_objects("model (1).h5")

# Define a function to make predictions
def predict_image(image_path):
    pred_image = Image.open(image_path).resize((224, 224))
    pred_image = np.array(pred_image) / 255.0
    pred_image = np.expand_dims(pred_image, axis=0)
    result = model.predict(pred_image)
    max_index = np.argmax(result)
    return max_index

# Streamlit app
st.title("Skin disease classification")

# Upload an image for prediction
uploaded_image = st.file_uploader("Upload an image ", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        try:
            max_index = predict_image(uploaded_image)
            classes = [
                'Hair Loss Photos Alopecia and other Hair Diseases',
                'Melanoma Skin Cancer Nevi and Moles',
                'Nail Fungus and other Nail Disease',
                'Urticaria Hives',
                'Warts Molluscum and other Viral Infections'
    
            ]
            st.write("Prediction:", classes[max_index])
        except Exception as e:
            st.error("An error occurred during prediction.")

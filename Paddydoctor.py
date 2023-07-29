import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model(r"C:\Final Year Project\Saved Models\Models Split(8.1.1)\MobileNet\MobileNet Model_best.h5")

# Define the class labels
class_labels = ['bacterial_leaf_blight', 'bacterial_leaf_streak', 
                'bacterial_panicle_blight', 'blast', 'brown_spot', 
                'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']  # Add your class labels here

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((128, 128))  # Resize the image to match the input size of the model
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = tf.keras.applications.mobilenet.preprocess_input(img_array)
    return preprocessed_img

# Function to make predictions
def make_prediction(image):
    preprocessed_img = preprocess_image(image)
    prediction = model.predict(preprocessed_img)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

# Streamlit app
st.title("Paddy Disease Classification Prototype")
st.write("Upload an image for classification")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Make predictions on uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    predicted_class, confidence = make_prediction(image)
    st.write("Predicted Class:", predicted_class)
    st.write("Confidence:", confidence)

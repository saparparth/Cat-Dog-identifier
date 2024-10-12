import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = load_model('cat_dog_classifier.h5')

# Set up the page
st.title("Cat vs Dog Image Classifier")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize image data

    # Make prediction
    prediction = model.predict(img_array)
    
    # Show the prediction result
    if prediction < 0.5:
        st.write("It's a Cat!")
    else:
        st.write("It's a Dog!")

    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

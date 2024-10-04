import os
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# Set custom page configuration
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="🧥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Define model path
model_path = 'app/trained_model/trained_fashion_mnist_model.h5'  # Ensure this path is correct

# Function to load the model
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model(model_path)

# Check if the model is loaded correctly
if model is None:
    st.error("Model could not be loaded. Please check the model path and try again.")

# Define class labels for the Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Function to preprocess the uploaded image
def preprocess_image(image):
    try:
        img = Image.open(image)
        img = img.resize((28, 28))  # Resize to 28x28
        img = img.convert('L')  # Convert to grayscale
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = img_array.reshape((1, 28, 28, 1))  # Reshape for model input
        return img_array
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        return None

# Custom advanced CSS for styling
st.markdown(
    """
    <style>
    /* General style for the app */
    body {
        background-color: white;
        font-family: 'Arial', sans-serif;
        color: black;
    }

    /* Custom button style */
    .stButton button {
        background-color: #4CAF50; /* Green button background */
        color: white;
        border: 2px solid #3e8e41; /* Darker green border */
        border-radius: 12px;
        padding: 12px;
        width: 100%;
        font-size: 18px;
        transition: all 0.3s ease;
    }

    /* Hover effect for buttons */
    .stButton button:hover {
        background-color: #45a049; /* Lighter green on hover */
        border-color: #367c3c; /* Darker border on hover */
        transform: scale(1.05);
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
    }

    /* Custom header style */
    h1 {
        color: #4CAF50; /* Green header color */
        text-align: center;
        margin-bottom: 20px;
        font-size: 2.5rem;
        animation: fadeIn 2s;
    }

    /* Image border styling */
    .stImage img {
        border: 3px solid #4CAF50; /* Green image border */
        border-radius: 15px;
        transition: transform 0.2s;
    }

    .stImage img:hover {
        transform: scale(1.05);
    }

    /* Animation for header */
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }

    /* Custom footer style */
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #4CAF50; /* Footer background color */
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 2px solid #367c3c; /* Darker border on top */
    }

    /* Sidebar styles */
    .sidebar .sidebar-content {
        background-color: #f5f5f5; /* Light background for sidebar */
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True
)

# App title with an animated effect
st.markdown("<h1>🧥 Fashion MNIST Classifier</h1>", unsafe_allow_html=True)

# Sidebar instructions
st.sidebar.markdown("<h3>Navigation</h3>", unsafe_allow_html=True)
st.sidebar.info(
    """
    - Upload an image in JPG, PNG, or JPEG format.
    - Click 'Classify Fashion Item' to see the result.
    - The classifier can predict items like T-shirts, sneakers, bags, etc.
    """
)

# File uploader widget with advanced design
uploaded_image = st.file_uploader("🔼 Upload a Fashion Item Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display original and preprocessed images side by side
    st.subheader("Original and Preprocessed Image")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Image**")
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.markdown("**Preprocessed (28x28)**")
        resized_img = image.resize((28, 28))
        st.image(resized_img, caption="Resized for Model Input", width=150)

    # Custom classify button with animations
    if st.button('🧠 Classify Fashion Item'):
        img_array = preprocess_image(uploaded_image)

        if img_array is not None and model is not None:
            with st.spinner('Processing...'):
                result = model.predict(img_array)
                predicted_class = np.argmax(result)
                confidence = np.max(result)

            # Display the prediction result and confidence score
            st.success(f'🛍️ Prediction: **{class_names[predicted_class]}**')
            st.info(f'📊 Confidence: **{confidence:.2f}**')

            # Enhance the user experience with animations
            st.balloons()
        else:
            st.error('Image could not be processed for prediction or model is not loaded.')
else:
    st.info("Please upload an image to start classification.")

# Footer
st.markdown(
    """
    <div class="footer">
        Developed by Your Name | Fashion MNIST Classifier App
    </div>
    """, unsafe_allow_html=True
)

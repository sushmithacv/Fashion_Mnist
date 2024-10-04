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

# Define paths and load the pre-trained model
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, 'trained_model', 'trained_fashion_mnist_model.h5')
print(f"Model path: {model_path}")  # Debugging line

# Function to load the model
def load_model():
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except OSError as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()  # Load the model at the start

# Define class labels for the Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Function to preprocess the uploaded image
def preprocess_image(image):
    """
    Preprocess the input image for prediction:
    - Convert to grayscale
    - Resize to 28x28 pixels
    - Normalize pixel values
    """
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
        background-color: #f8f9fa;
        font-family: 'Arial', sans-serif;
        color: #343a40; /* Dark gray for text */
    }

    /* Button style */
    .stButton button {
        background-color: #007bff; /* Bootstrap primary color */
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px;
        width: 100%;
        font-size: 18px;
        transition: all 0.3s ease;
    }

    /* Hover effect for buttons */
    .stButton button:hover {
        background-color: #0056b3; /* Darker shade on hover */
        transform: scale(1.05);
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
    }

    /* Header style */
    h1 {
        color: #007bff; /* Bootstrap primary color */
        text-align: center;
        margin-bottom: 20px;
        font-size: 2.5rem;
    }

    /* Image border styling */
    .stImage img {
        border: 5px solid #007bff;
        border-radius: 8px;
        transition: transform 0.2s;
    }

    .stImage img:hover {
        transform: scale(1.05);
    }

    /* Footer style */
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #007bff;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 2px solid #0056b3;
    }
    </style>
    """, unsafe_allow_html=True
)

# App title
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

# File uploader widget
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

    # Classify button
    if st.button('🧠 Classify Fashion Item'):
        if model is not None:  # Check if the model is loaded successfully
            img_array = preprocess_image(uploaded_image)

            if img_array is not None:
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
                st.error('Image could not be processed for prediction.')
        else:
            st.error("The model is not loaded. Please check the logs for more information.")
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

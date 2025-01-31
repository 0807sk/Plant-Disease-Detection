import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load Model Once (Prevents Repeated Loading)
MODEL_PATH = 'E:/Sanjeev/Learnings/Projects/ML/Plant Disease Detection/Trained_model.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# Function to Predict Disease
def model_prediction(image):
    image = image.resize((128, 128))  # Resize to model input shape
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    predictions = model.predict(image)
    result_index = np.argmax(predictions)  # Get index of highest probability class
    return result_index

# Sidebar Navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("🌿 Plant Disease Detection System")
    image_path = "images.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    This **Plant Disease Detection System** is a **Streamlit**-based web application that enables users to upload an image of a plant leaf and get predictions on whether the plant is diseased or healthy.

    ## 🚀 Features
    - Upload an image of a plant leaf.
    - Get a real-time disease detection result.
    - Uses a pre-trained **Deep Learning (CNN) model** for classification.
    """)

# About Page
elif app_mode == "About":
    st.header("🌿 About Plant Disease Detection System")
    st.markdown("""
    ### 📊 Dataset Information:
    - **Train:** 70,295 Images  
    - **Validation:** 17,572 Images  
    - **Test:** 33 Images
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("🌿 Plant Disease Detection")

    # File Upload
    uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)  # Convert to PIL Image
        if st.button("Show Image"):
            st.image(image, use_column_width=True)
        
        # Predict Button
        if st.button("Predict"):
            st.write("🔍 **Model Prediction:**")
            result_index = model_prediction(image)

            # Define Class Labels
            class_names = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]
            
            disease_name = class_names[result_index]
            st.success(f"✅ **Prediction:** {disease_name}")

import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Set page config (This MUST be the first Streamlit command)
st.set_page_config(
    page_title="Multiple Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Create necessary directories
for dir_path in ['uploads']:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

#-----------------
# Image Validation Functions
#-----------------

# Function to validate retinal/eye images with improved recognition for various imaging types
def validate_eye_image(img):
    """
    Validate if the uploaded image appears to be an eye or retinal image,
    accounting for different eye imaging techniques (including fluorescein and other stains)
    """
    # Convert PIL image to cv2 format
    img_cv = np.array(img.convert('RGB'))
    img_cv = img_cv[:, :, ::-1].copy()  # RGB to BGR
    
    # Check image characteristics
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 1. Check for circular shape (common in eye images)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
        param1=50, param2=30, minRadius=50, maxRadius=400
    )
    has_circular_features = circles is not None
    
    # 2. Examine color characteristics for various eye imaging types
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    
    # Traditional retinal scans (reddish-orange)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([30, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    red_pixel_ratio = np.sum(mask_red > 0) / (img_cv.shape[0] * img_cv.shape[1])
    
    # Fluorescein angiography or other staining (greenish-blue)
    lower_green_blue = np.array([75, 50, 50])
    upper_green_blue = np.array([150, 255, 255])
    mask_green_blue = cv2.inRange(hsv, lower_green_blue, upper_green_blue)
    green_blue_pixel_ratio = np.sum(mask_green_blue > 0) / (img_cv.shape[0] * img_cv.shape[1])
    
    # 3. Check for iris-like texture patterns
    # Apply edge detection to identify detailed texture patterns common in eye images
    edges = cv2.Canny(blurred, 50, 150)
    edge_density = np.sum(edges > 0) / (img_cv.shape[0] * img_cv.shape[1])
    has_detailed_texture = edge_density > 0.05  # Threshold for detailed texture
    
    # 4. Check brightness distribution (eye images often have a characteristic light distribution)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    bright_ratio = np.sum(binary > 0) / (img_cv.shape[0] * img_cv.shape[1])
    has_eye_brightness = 0.2 < bright_ratio < 0.8  # Not too dark, not too bright
    
    # Combined validation criteria - accept if it meets the circular criterion plus color OR texture patterns
    has_eye_colors = red_pixel_ratio > 0.1 or green_blue_pixel_ratio > 0.1
    
    # More permissive validation - if it has eye-like features
    return (has_circular_features and (has_eye_colors or has_detailed_texture)) or \
           (has_detailed_texture and has_eye_brightness and (red_pixel_ratio > 0.05 or green_blue_pixel_ratio > 0.05))
# Function to validate chest X-ray images
def validate_xray_image(img):
    """
    Validate if the uploaded image appears to be a chest X-ray
    using image processing features like grayscale characteristics and aspect ratio
    """
    # Convert PIL image to cv2 format
    img_cv = np.array(img.convert('RGB'))
    img_cv = img_cv[:, :, ::-1].copy()  # RGB to BGR
    
    # 1. Check grayscale characteristics - X-rays appear mostly grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    rgb_mean = np.mean(img_cv, axis=2)
    rgb_std = np.std(img_cv, axis=2)
    
    # Calculate the average standard deviation across color channels
    # Low standard deviation means the image is closer to grayscale
    color_std = np.mean(rgb_std)
    
    # 2. Check histogram distribution (X-rays have specific histogram patterns)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist / hist.sum()
    
    # 3. Check contrast - X-rays typically have good contrast
    min_val, max_val, _, _ = cv2.minMaxLoc(gray)
    contrast = max_val - min_val
    
    # Criteria for validation
    is_grayscale_like = color_std < 30
    has_good_contrast = contrast > 50
    
    # Additional check for histogram distribution characteristic of X-rays
    # X-rays typically have peaks in the middle-range of grayscale
    mid_range_concentration = np.sum(hist_norm[50:200]) > 0.5
    
    return is_grayscale_like and has_good_contrast and mid_range_concentration

#-----------------
# Eye Disease Functions
#-----------------

# Function to load eye disease model
@st.cache_resource
def load_eye_prediction_model():
    model_path = 'eye_disease/models/eye_disease_model.h5'
    if os.path.exists(model_path):
        model = load_model(model_path)
        return model
    return None

# Function to preprocess the eye image
def preprocess_eye_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make eye disease prediction
def predict_eye_disease(model, img_array):
    prediction = model.predict(img_array)
    return prediction

# Function to display eye prediction results
def display_eye_prediction(prediction, category_names):
    category_idx = np.argmax(prediction[0])
    category_confidence = prediction[0][category_idx] * 100
    
    st.subheader("Prediction Results")
    st.markdown(f"**Predicted Category:** {category_names[category_idx]}")
    st.markdown(f"**Confidence:** {category_confidence:.2f}%")
    
    st.subheader("Condition Information")
    
    condition_info = {
        0: {
            "name": "Normal",
            "description": "The eye appears healthy with no signs of disease.",
            "treatment": "Regular eye check-ups recommended for preventive care."
        },
        1: {
            "name": "Diabetic Retinopathy",
            "description": "A diabetes complication that affects the eyes, caused by damage to blood vessels in the retina.",
            "treatment": "Treatment options include laser treatment, anti-VEGF therapy, and vitrectomy depending on severity."
        },
        2: {
            "name": "Glaucoma",
            "description": "A group of eye conditions that damage the optic nerve, often caused by abnormally high pressure in the eye.",
            "treatment": "Treatments include eye drops, oral medications, laser treatment, and surgery to reduce intraocular pressure."
        },
        3: {
            "name": "Cataract",
            "description": "A clouding of the normally clear lens of the eye, leading to decreased vision.",
            "treatment": "Cataract surgery is the most effective treatment, replacing the cloudy lens with an artificial one."
        },
        4: {
            "name": "Age-related Macular Degeneration",
            "description": "A eye disease that causes vision loss in the center of the field of vision.",
            "treatment": "Treatment options include anti-VEGF drugs, photodynamic therapy, and laser therapy, depending on the type and stage."
        }
    }
    
    if category_idx in condition_info:
        info = condition_info[category_idx]
        st.markdown(f"**Condition:** {info['name']}")
        st.markdown(f"**Description:** {info['description']}")
        st.markdown(f"**Possible Treatment:** {info['treatment']}")
    
    st.warning("Note: This prediction is for informational purposes only and should not replace professional medical advice. Please consult a healthcare professional for proper diagnosis and treatment.")

#-----------------
# COPD Functions
#-----------------

# Function to make COPD predictions
def predict_copd(img, model):
    # Convert to numpy array
    img_array = np.array(img)
    
    # Check if the image is grayscale and convert to RGB if needed
    if len(img_array.shape) == 2:
        # If it's a 2D array (height, width), convert to 3D with 3 channels
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 1:
        # Convert from grayscale to RGB by duplicating the channel
        img_array = np.concatenate([img_array] * 3, axis=-1)
    
    # Ensure we have 3 channels
    if img_array.shape[-1] != 3:
        st.error(f"Unexpected image shape: {img_array.shape}. Converting to RGB.")
        # Convert to RGB as a fallback
        img_pil = Image.fromarray(img_array.astype('uint8'))
        img_pil = img_pil.convert('RGB')
        img_array = np.array(img_pil)
    
    # Normalize
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)
    
    return prediction[0][0]

# Function to load the COPD model
@st.cache_resource
def load_copd_model():
    model_path = 'lungsmodel/copd_detection_model.h5'
    if os.path.exists(model_path):
        return load_model(model_path)
    return None

#-----------------
# Main Application
#-----------------

def main():
    # Initialize session state for navigation if it doesn't exist
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = "Home"
    
    # Title and introduction
    st.title("Multiple Disease Prediction System")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    # Use selectbox with the current value from session state
    app_mode = st.sidebar.selectbox(
        "Select Disease Prediction System", 
        ["Home", "Eye Disease Prediction", "COPD Detection"],
        index=["Home", "Eye Disease Prediction", "COPD Detection"].index(st.session_state.app_mode)
    )
    
    # Update session state when sidebar selection changes
    st.session_state.app_mode = app_mode
    
    # Sidebar about section (common across all pages)
    st.sidebar.header("About")
    st.sidebar.info("""
    This unified medical prediction system uses deep learning models to assist in preliminary 
    diagnosis of various diseases. Each model has been trained on thousands of medical images.
    
    All predictions are for informational purposes only and should be confirmed by medical professionals.
    """)
    
    # Home page
    if app_mode == "Home":
        st.markdown("""
        ### Welcome to the Unified Medical Diagnosis Assistant
        
        This application integrates multiple disease prediction models to assist in preliminary medical screening.
        
        **Currently Available Models:**
        - **Eye Disease Prediction**: Detects various eye conditions from retinal images
        - **COPD Detection**: Analyzes chest X-rays for signs of Chronic Obstructive Pulmonary Disease
        
        **🚨 Disclaimer:** This tool is for educational and screening purposes only. 
        It is not intended to replace professional medical diagnosis. Always consult with healthcare professionals.
        """)
        
        # Two columns for the two models
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Eye Disease Prediction")
            # Create a placeholder image instead of using an empty string
            st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRXO9K9VCBtF4UYt4YFh1cifaeqwxU1ZUYfWw&s", use_column_width=True)
            st.markdown("""
            Upload retinal images to detect conditions like:
            - Diabetic Retinopathy
            - Glaucoma
            - Cataract
            - Age-related Macular Degeneration
            """)
            if st.button("Go to Eye Disease Prediction", key="eye_btn"):
                st.session_state.app_mode = "Eye Disease Prediction"
                st.rerun()
        
        with col2:
            st.subheader("COPD Detection")
            st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTjstXBJyH046igwIpzWbUkKy5UO9DaIY8Www&s", use_column_width=True)
            st.markdown("""
            Upload chest X-ray images to detect:
            - Chronic Obstructive Pulmonary Disease (COPD)
            
            The model analyzes patterns in the lungs that may indicate COPD.
            """)
            if st.button("Go to COPD Detection", key="copd_btn"):
                st.session_state.app_mode = "COPD Detection"
                st.rerun()
                
        # Additional information
        st.header("How It Works")
        st.markdown("""
        1. **Select a prediction model** from the sidebar
        2. **Upload a medical image** relevant to the selected model
        3. **Get instant prediction** based on deep learning analysis
        4. **Consult with healthcare professionals** for proper diagnosis
        
        All models use state-of-the-art convolutional neural networks trained on extensive medical image datasets.
        """)
    
    # Eye Disease Prediction page
    elif app_mode == "Eye Disease Prediction":
        st.header("Eye Disease Prediction")
        st.markdown("""
        ### Introduction
        This application uses deep learning to predict eye diseases from uploaded retinal images. 
        The model uses a pre-trained MobileNetV2 architecture to efficiently classify different eye conditions.

        **Types of Eye Diseases the Model Can Detect:**
        - Diabetic Retinopathy
        - Glaucoma
        - Cataract
        - Age-related Macular Degeneration
        - Normal

        Upload a clear retinal image to get a prediction of the eye condition.
        """)
        
        # Add button to return to home
        if st.button("Return to Home", key="eye_return_btn"):
            st.session_state.app_mode = "Home"
            st.rerun()
        
        # Define eye disease categories
        category_dict = {0: "Normal", 1: "Diabetic Retinopathy", 2: "Glaucoma", 
                       3: "Cataract", 4: "Age-related Macular Degeneration"}
        
        # Load eye model
        model = load_eye_prediction_model()
        
        # If model exists, show prediction section
        if model is not None:
            predict_file = st.file_uploader("Upload an image for prediction", type=["jpg", "jpeg", "png"], key="eye_uploader")
            
            if predict_file:
                # Display uploaded image
                img = Image.open(predict_file)
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Uploaded Image")
                    st.image(img, width=300, caption="Uploaded Image")
                
                # First validate if it's a proper retinal image
                is_valid_eye_image = validate_eye_image(img)
                
                if not is_valid_eye_image:
                    st.error("⚠️ The uploaded image does not appear to be a valid retinal/eye scan image. Please upload a proper retinal scan image for accurate prediction.")
                    st.info("Retinal images typically have circular boundaries and reddish-orange coloration. Make sure you're uploading a medical-grade retinal scan.")
                else:
                    # Process image and predict
                    with col2:
                        with st.spinner("Analyzing image..."):
                            img = img.convert('RGB')  # Ensure RGB format
                            img_array = preprocess_eye_image(img)
                            
                            # Make prediction
                            prediction = predict_eye_disease(model, img_array)
                            display_eye_prediction(prediction, category_dict)
        else:
            st.error("Eye disease model not found. Please make sure you have the model file at 'eye_disease/models/eye_disease_model.h5'.")
    
    # COPD Detection page
    elif app_mode == "COPD Detection":
        st.header("COPD Detection from Chest X-rays")
        st.markdown("""
        ### Introduction
        This application uses a Convolutional Neural Network (CNN) to detect Chronic Obstructive Pulmonary Disease (COPD) 
        from chest X-ray images. Upload an X-ray image to get a prediction.
        """)
        
        # Add button to return to home
        if st.button("Return to Home", key="copd_return_btn"):
            st.session_state.app_mode = "Home"
            st.rerun()
        
        # Load COPD model
        model = load_copd_model()
        
        # File uploader for COPD detection
        uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"], key="xray_uploader")
        
        if uploaded_file is not None:
            if model is not None:
                # Display original image
                image_pil = Image.open(uploaded_file)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Uploaded Image")
                    st.image(image_pil, caption="Uploaded Image", use_column_width=True)
                
                # First validate if it's a proper X-ray image
                is_valid_xray = validate_xray_image(image_pil)
                
                if not is_valid_xray:
                    st.error("⚠️ The uploaded image does not appear to be a valid chest X-ray. Please upload a proper chest X-ray image for accurate prediction.")
                    st.info("Chest X-rays are typically grayscale images with specific contrast patterns showing lung and ribcage structures.")
                else:
                    # Resize image for prediction
                    img = image_pil.resize((150, 150))
                    
                    # Make prediction
                    with st.spinner("Analyzing image..."):
                        prediction_value = predict_copd(img, model)
                    
                    # Display results
                    with col2:
                        st.subheader("Prediction Results")
                        
                        # Create a figure for matplotlib
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.imshow(np.array(img))
                        
                        # Determine if positive or negative
                        threshold = 0.95
                        
                        if prediction_value > threshold:
                            result = f'Positive: {prediction_value:.2f}'
                            confidence = prediction_value * 100
                            color = 'red'
                            st.error(f"Positive COPD Detection (Confidence: {prediction_value:.2f})")
                            st.warning("Please consult with a healthcare professional for proper diagnosis.")
                        else:
                            result = f'Negative: {(1.0 - prediction_value):.2f}'
                            confidence = (1.0 - prediction_value) * 100
                            color = 'green'
                            st.success(f"Negative COPD Detection (Confidence: {(1.0 - prediction_value):.2f})")
                            st.info("Note: This is an automated prediction and should not replace professional medical advice.")
                        
                        # Add text to the image
                        ax.text(30, 145, result, color=color, fontsize=18, 
                                bbox=dict(facecolor='white', alpha=0.8))
                        
                        # Remove axis
                        ax.axis('off')
                        
                        # Display the plot in Streamlit
                        st.pyplot(fig)
                        
                        # Display confidence as a progress bar
                        st.markdown(f"**Confidence: {confidence:.2f}%**")
                        st.progress(int(confidence))
            else:
                st.error("COPD model not found. Please make sure you have the model file at 'lungsmodel/copd_detection_model.h5'.")
        else:
            # Display sample X-ray if no file is uploaded
            st.info("Please upload an X-ray image to get a prediction.")
            st.image("https://via.placeholder.com/300x300.png?text=Sample+Chest+X-ray", caption="Sample X-ray Image")

# Run the app
if __name__ == "__main__":
    main()
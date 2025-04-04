import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import gc

# Set page config
st.set_page_config(
    page_title="Eye Disease Prediction",
    page_icon="👁️",
    layout="wide"
)

# Create necessary directories
for dir_path in ['uploads']:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Title and introduction
st.title("Eye Disease Prediction App")
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

# Function to load model
@st.cache_resource
def load_prediction_model():
    model_path = 'models/eye_disease_model.h5'
    if os.path.exists(model_path):
        model = load_model(model_path)
        return model
    return None

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make prediction
def predict_disease(model, img_array):
    prediction = model.predict(img_array)
    return prediction

# Function to display prediction results
def display_prediction(prediction, category_names):
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

# Main application flow
def main():
    # Define paths
    labels_path = "dataset/labels.csv"
    
    # Sidebar
    st.sidebar.header("Application Control")
    
    # Model prediction section
    model = load_prediction_model()
    
    # Load categories from labels.csv
    try:
        if os.path.exists(labels_path):
            df = pd.read_csv(labels_path)
            category_dict = {}
            
            for idx, category in enumerate(df['category'].unique()):
                category_dict[idx] = category
        else:
            category_dict = {0: "Normal", 1: "Diabetic Retinopathy", 2: "Glaucoma", 
                          3: "Cataract", 4: "Age-related Macular Degeneration"}
    except Exception as e:
        st.sidebar.error(f"Error loading categories: {e}")
        category_dict = {0: "Normal", 1: "Diabetic Retinopathy", 2: "Glaucoma", 
                      3: "Cataract", 4: "Age-related Macular Degeneration"}
    
    # If model exists, show prediction section
    if model is not None:
        predict_file = st.file_uploader("Upload an image for prediction", type=["jpg", "jpeg", "png"])
        
        if predict_file:
            # Display uploaded image
            img = Image.open(predict_file)
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Uploaded Image")
                st.image(img, width=300, caption="Uploaded Retinal Image")
            
            # Process image and predict
            with col2:
                with st.spinner("Analyzing image..."):
                    img = img.convert('RGB')  # Ensure RGB format
                    img_array = preprocess_image(img)
                    
                    # Make prediction
                    prediction = predict_disease(model, img_array)
                    display_prediction(prediction, category_dict)
    else:
        st.error("Model not found. Please make sure you have the model file at 'models/eye_disease_model.h5'.")
    
    # About section
    st.sidebar.header("About")
    st.sidebar.info("""
    This application uses transfer learning with MobileNetV2 to efficiently predict eye diseases from retinal images.
    
    Remember that this tool is for educational purposes only and should not replace professional medical advice.
    """)

# Run the app
if __name__ == "__main__":
    main()
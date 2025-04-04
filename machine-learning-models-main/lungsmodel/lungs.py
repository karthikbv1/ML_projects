import os
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import io
from PIL import Image
import tempfile

# Set page configuration
st.set_page_config(
    page_title="COPD Detection App",
    page_icon="🫁",
    layout="wide"
)

# Constants
IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150

# Function to make predictions
def predict_copd(img, model):
    # Convert to numpy array
    img_array = image.img_to_array(img)
    
    # Check if the image is grayscale and convert to RGB if needed
    if img_array.shape[-1] == 1:
        # Convert from grayscale to RGB by duplicating the channel
        img_array = np.concatenate([img_array] * 3, axis=-1)
    elif len(img_array.shape) == 2:
        # If it's a 2D array (height, width), convert to 3D with 3 channels
        img_array = np.stack([img_array] * 3, axis=-1)
    
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
    
    # Debugging information
    st.sidebar.write(f"Processed image shape: {img_array.shape}")
    
    # Make prediction
    prediction = model.predict(img_array, batch_size=None, steps=1)
    
    return prediction[0][0]

def main():
    # App title and description
    st.title("COPD Detection from Chest X-rays")
    st.markdown("""
    This application uses a Convolutional Neural Network (CNN) to detect Chronic Obstructive Pulmonary Disease (COPD) 
    from chest X-ray images. Upload an X-ray image to get a prediction.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    pages = ["Home", "About", "How It Works"]
    selection = st.sidebar.radio("Go to", pages)
    
    # Home page - Upload and predict
    if selection == "Home":
        st.header("Upload X-ray Image")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
        
        col1, col2 = st.columns(2)
        
        if uploaded_file is not None:
            # Load the model
            @st.cache_resource
            def load_cnn_model():
                return load_model('copd_detection_model.h5')
            
            try:
                model = load_cnn_model()
                
                # Display original image
                with col1:
                    st.subheader("Uploaded Image")
                    image_pil = Image.open(uploaded_file)
                    
                    # Display image info in sidebar for debugging
                    st.sidebar.write("Image Information:")
                    st.sidebar.write(f"Original size: {image_pil.size}")
                    st.sidebar.write(f"Mode: {image_pil.mode}")
                    
                    # Convert to RGB if not already
                    if image_pil.mode != 'RGB':
                        st.sidebar.write(f"Converting from {image_pil.mode} to RGB")
                        image_pil = image_pil.convert('RGB')
                    
                    st.image(image_pil, caption="Uploaded X-ray", use_column_width=True)
                
                # Resize image for prediction
                img = image_pil.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
                
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
                    threshold = 0.95  # Using the same threshold as in your notebook
                    
                    if prediction_value > threshold:
                        result = f'Positive: {prediction_value:.2f}'
                        confidence = prediction_value * 100
                        color = 'red'
                    else:
                        result = f'Negative: {(1.0 - prediction_value):.2f}'
                        confidence = (1.0 - prediction_value) * 100
                        color = 'green'
                    
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
                    
                    # Additional explanation
                    if prediction_value > threshold:
                        st.error("The model detected signs of COPD in this X-ray.")
                        st.warning("Please consult with a healthcare professional for proper diagnosis.")
                    else:
                        st.success("The model did not detect signs of COPD in this X-ray.")
                        st.info("Note: This is an automated prediction and should not replace professional medical advice.")
            
            except Exception as e:
                st.error(f"Error loading model or processing image: {e}")
                st.error("Please make sure you have the trained model 'copd_detection_model.h5' in the application directory.")
                st.info("Detailed traceback:")
                import traceback
                st.code(traceback.format_exc())
        
        else:
            # Display sample X-ray if no file is uploaded
            st.info("Please upload an X-ray image to get a prediction.")
            st.image("https://via.placeholder.com/300x300.png?text=Sample+Chest+X-ray", caption="Sample X-ray Image")
    
    # About page
    elif selection == "About":
        st.header("About COPD Detection")
        st.markdown("""
        ### What is COPD?
        
        Chronic Obstructive Pulmonary Disease (COPD) is a chronic inflammatory lung disease that causes obstructed airflow from the lungs. 
        Symptoms include breathing difficulty, cough, mucus production and wheezing.
        
        ### About This Application
        
        This application uses a deep learning Convolutional Neural Network (CNN) model trained on thousands of X-ray images 
        to detect signs of COPD in chest X-rays. The model was trained on a dataset of 3000 images with an 80-20 train-test split.
        
        ### Important Disclaimer
        
        This application is for educational and demonstration purposes only. It should not be used for actual medical diagnosis. 
        Always consult with qualified healthcare professionals for medical advice and diagnosis.
        """)
    
    # How It Works page
    elif selection == "How It Works":
        st.header("How the COPD Detection Works")
        st.markdown("""
        ### Deep Learning Model
        
        This application uses a Convolutional Neural Network (CNN) to analyze chest X-ray images:
        
        1. **Preprocessing**: When you upload an image, it's resized to 150x150 pixels and normalized.
        
        2. **Model Architecture**: The CNN consists of:
           - 4 convolutional layers with max pooling
           - Flatten layer
           - Dense layer with dropout for regularization
           - Output layer with sigmoid activation for binary classification
        
        3. **Prediction**: The model assigns a probability score indicating the likelihood of COPD presence.
        
        ### Model Training
        
        The model was trained on a dataset of 3000 chest X-ray images:
        - 2400 images used for training (80%)
        - 600 images used for testing (20%)
        
        The training process included data augmentation techniques like rotation, zoom, and horizontal flipping to improve the model's generalization capability.
        """)
        
        # Display model architecture
        st.subheader("Model Architecture")
        model_arch = """
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        """
        st.code(model_arch, language="python")

if __name__ == "__main__":
    main()
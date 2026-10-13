import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import json
import os
from model import preprocess_image

# Page configuration
st.set_page_config(page_title="Sign Language Recognition", layout="wide")

# Load model and class indices
@st.cache_resource
def load_model():
    model_path = 'models/sign_language_model.h5'
    class_indices_path = 'models/class_indices.json'
    
    if not os.path.exists(model_path):
        st.error("Model not found. Please train the model first by running train.py")
        return None, None
    
    model = tf.keras.models.load_model(model_path)
    
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    
    # Create reverse mapping (index -> class name)
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    return model, idx_to_class

def predict_image(model, idx_to_class, image):
    """
    Predict the sign language from an image.
    
    Args:
        model: Loaded Keras model
        idx_to_class: Mapping from index to class name
        image: PIL Image
        
    Returns:
        Predicted class and confidence
    """
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize and normalize
    img_resized = cv2.resize(img_bgr, (128, 128))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype('float32') / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Predict
    predictions = model.predict(img_batch, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    predicted_class = idx_to_class[predicted_idx]
    
    return predicted_class, confidence

def main():
    st.title("Sign Language Recognition")
    st.write("Upload an image of a hand sign to recognize the ASL letter.")
    
    # Load model
    model, idx_to_class = load_model()
    
    if model is None:
        st.stop()
    
    # Initialize session state for accumulated text
    if 'accumulated_text' not in st.session_state:
        st.session_state.accumulated_text = ""
    
    # Two columns: upload and results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Predict
            with st.spinner("Recognizing sign..."):
                predicted_class, confidence = predict_image(
                    model, idx_to_class, image
                )
            
            # Display prediction
            st.success(f"Predicted Sign: **{predicted_class}**")
            st.info(f"Confidence: **{confidence:.2%}**")
            
            # Add to accumulated text
            if st.button(f"Add '{predicted_class}' to text"):
                st.session_state.accumulated_text += predicted_class
                st.rerun()
    
    with col2:
        st.subheader("Accumulated Text")
        text_area = st.text_area(
            "Recognized signs will appear here:",
            value=st.session_state.accumulated_text,
            height=200
        )
        
        if st.button("Clear Text"):
            st.session_state.accumulated_text = ""
            st.rerun()
        
        # Webcam section (optional)
        st.subheader("Live Webcam Recognition")
        st.warning("Webcam support requires additional setup. See README for details.")
        
        use_webcam = st.checkbox("Enable Webcam (Experimental)")
        
        if use_webcam:
            st.write("Click the button below to capture from webcam")
            if st.button("Capture Frame"):
                # Note: This is a simplified version
                # Full webcam integration requires additional handling
                try:
                    cap = cv2.VideoCapture(0)
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image_pil = Image.fromarray(frame_rgb)
                        st.image(image_pil, caption="Webcam Capture", use_column_width=True)
                        
                        with st.spinner("Recognizing sign..."):
                            predicted_class, confidence = predict_image(
                                model, idx_to_class, image_pil
                            )
                        
                        st.success(f"Predicted Sign: **{predicted_class}**")
                        st.info(f"Confidence: **{confidence:.2%}**")
                        
                        if st.button(f"Add '{predicted_class}' to text (Webcam)"):
                            st.session_state.accumulated_text += predicted_class
                            st.rerun()
                    else:
                        st.error("Could not capture from webcam")
                except Exception as e:
                    st.error(f"Webcam error: {str(e)}")

if __name__ == '__main__':
    main()
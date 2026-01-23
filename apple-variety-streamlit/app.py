import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import time

# Page Configuration
st.set_page_config(
    page_title="Apple Variety Classifier",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Determine the directory of the current file (app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Constants - using absolute paths based on BASE_DIR
MODEL_PATH = os.path.join(BASE_DIR, "model", "apple_classifier_final.keras")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")
STYLES_PATH = os.path.join(BASE_DIR, "assets", "styles.css")
CONFIDENCE_THRESHOLD = 0.75

# --- Utils ---
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"Style file not found: {file_name}")

@st.cache_resource
def load_model_cached():
    """Loads the model with caching to avoid reloading on every interaction."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_labels():
    """Loads the class labels."""
    if not os.path.exists(LABELS_PATH):
        return {}
    with open(LABELS_PATH, 'r') as f:
        return json.load(f)

def preprocess_image(image):
    """
    Preprocesses the image for EfficientNetV2.
    1. Resize to 224x224
    2. Convert to array
    3. Expand dims (1, 224, 224, 3)
    4. Preprocess (EfficientNetV2 expects 0-255 inputs usually, but built-in func handles it)
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    
    # EfficientNetV2S preprocessing
    # If using keras.applications, we can use the specific preprocess_input
    # But to stay lightweight, we can just ensure it's correct.
    # TF efficientnet_v2.preprocess_input typically passes through 0-255.
    # However, good practice to use the official utility if available.
    
    return tf.keras.applications.efficientnet_v2.preprocess_input(img_array)

# --- Main App ---

# Inject CSS
# Inject CSS
local_css(STYLES_PATH)

# Header
st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1 style='font-weight: 700; color: #1d1d1f; margin-bottom: 10px;'>Apple Variety Classifier</h1>
        <p style='color: #86868b; font-size: 1.1rem;'>Professional Grade AI Assessment System</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/15/Red_Apple.jpg", width=100)
    st.title("Settings")
    
    st.markdown("### Model Details")
    st.info("""
    **Architecture**: EfficientNetV2S
    **Classes**: 10 Market Varieties
    **Accuracy**: ~96-98%
    """)
    
    show_gradcam = st.toggle("Explain Predictions (Coming Soon)", value=False, disabled=True)
    st.caption("Enable visualization layers (disabled in lite version)")
    
    st.markdown("---")
    st.caption("Built with TensorFlow & Streamlit")

# Main Content
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 1. Upload Image")
    uploaded_file = st.file_uploader("Drag and drop an apple image here", type=['jpg', 'jpeg', 'png', 'webp'])
    
    image_placeholder = st.empty()
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image_placeholder.image(image, caption="Uploaded Specimen", use_container_width=True)
        except Exception as e:
            st.error("Invalid image file.")
            image = None
    else:
        # Placeholder styling
        st.info("Awaiting image upload...")

with col2:
    st.markdown("### 2. Analysis")
    
    if uploaded_file is not None:
        classify_btn = st.button("Classify Apple Variety", type="primary")
        
        if classify_btn:
             # Load resources
            model = load_model_cached()
            labels = load_labels()
            
            if model is None:
                st.error(f"⚠️ Model file not found at `{MODEL_PATH}`. Please place the `.keras` model file in the directory.")
            elif not labels:
                st.error(f"⚠️ Labels file not found at `{LABELS_PATH}`.")
            else:
                with st.spinner("Analyzing texture and color patterns..."):
                    # Inference
                    try:
                        processed_img = preprocess_image(image)
                        start_time = time.time()
                        predictions = model.predict(processed_img)
                        inference_time = time.time() - start_time
                        
                        scores = tf.nn.softmax(predictions[0]) if predictions.shape[-1] > 1 else predictions[0] # Handle raw logits vs softmax if needed, but usually predict returns probs? 
                        # Notebook says "Softmax Output Layer". predict() usually returns the output. If output layer is Softmax, it's probs.
                        # We will assume it returns probabilities.
                        
                        top_k = 5
                        top_indices = np.argsort(predictions[0])[::-1][:top_k]
                        top_score = predictions[0][top_indices[0]]
                        top_class_idx = top_indices[0]
                        top_class_name = labels.get(str(top_class_idx), f"Class {top_class_idx}")
                        
                        # Display Results
                        confidence_percent = top_score * 100
                        
                        # Determine card style
                        card_class = "success" if top_score >= 0.9 else "warning" if top_score >= CONFIDENCE_THRESHOLD else "danger"
                        border_color = "#34C759" if top_score >= 0.9 else "#FFcc00" if top_score >= CONFIDENCE_THRESHOLD else "#FF3B30"
                        
                        st.markdown(f"""
                        <div class="prediction-card {card_class}" style="border-left-color: {border_color};">
                            <div class="class-name">Identified Variety</div>
                            <div class="confidence-score" style="color: {border_color};">{top_class_name}</div>
                            <div style="margin-top: 10px; color: #1d1d1f; font-weight: 500;">
                                Confidence: {confidence_percent:.1f}%
                            </div>
                            <div style="font-size: 0.8rem; color: #86868b; margin-top: 5px;">
                                Inference time: {inference_time*1000:.1f}ms
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if top_score < CONFIDENCE_THRESHOLD:
                            st.warning(f"⚠️ Low confidence detection (< {CONFIDENCE_THRESHOLD*100}%). This might not be one of the known 10 varieties.")
                        
                        st.markdown("#### Top Probabilities")
                        for idx in top_indices:
                            class_name = labels.get(str(idx), f"Class {idx}")
                            score = predictions[0][idx]
                            st.markdown(f"**{class_name}**")
                            st.progress(float(score))
                            st.caption(f"{score*100:.1f}%")
                            
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

    else:
        st.markdown("""
        <div style="padding: 40px; text-align: center; color: #86868b; background: #f9f9f9; border-radius: 12px;">
            Upload an image to start analysis
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("© 2025 DeepApple AI • Research Use Only")

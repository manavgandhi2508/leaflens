import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

def preprocess_image_for_model(image: Image.Image) -> np.ndarray:
    """
    Preprocess the uploaded image for prediction.
    Must strictly match the original model's preprocessing:
    - Convert to RGB
    - Resize to (128, 128)
    - Convert to array
    - Expand dimensions to (1, 128, 128, 3)
    - NO numerical scaling (e.g. / 255.0) as it wasn't in original code.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((128, 128))
    
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.array([img_array])  # Create batch dimension
    
    return img_array

def load_image_from_upload(uploaded_file) -> Image.Image:
    """Loads an image from a Streamlit uploaded file and sanitizes it."""
    image = Image.open(uploaded_file)
    
    # Safely strip EXIF data and apply rotation (prevents Streamlit TypeErrors on mobile images)
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass
        
    # Convert to standard RGB to prevent rendering errors in st.image
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    return image

import numpy as np
import tensorflow as tf
from PIL import Image

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
    """Loads an image from a Streamlit uploaded file."""
    return Image.open(uploaded_file)

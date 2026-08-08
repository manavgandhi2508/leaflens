import os
import numpy as np
import tensorflow as tf
import streamlit as st
from data.disease_info import CLASS_NAMES

@st.cache_resource
def load_cnn_model():
    """Load the trained model with Streamlit caching."""
    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "plant_disease_model.keras")
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def parse_class_name(predicted_class: str):
    """
    Parses the class name (e.g. 'Tomato___Early_blight') 
    into Plant, Disease, and Status.
    """
    if "___" in predicted_class:
        plant_type, disease = predicted_class.split("___", 1)
        plant_type = plant_type.replace("_", " ").replace(",", "").title()
        disease = disease.replace("_", " ").title()
        
        if disease.lower() == "healthy":
            status = "Healthy"
            disease = "No disease detected"
        else:
            status = "Diseased"
    else:
        plant_type = predicted_class.replace("_", " ").title()
        disease = "Unknown"
        status = "Unknown"
        
    return plant_type, disease, status

def predict_disease(model, img_array: np.ndarray) -> dict:
    """Runs inference and returns the top predictions."""
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = float(predictions[0][predicted_class_index])
    
    plant_type, disease, status = parse_class_name(predicted_class)
    
    # Get top 3 predictions
    top_indices = np.argsort(predictions[0])[-3:][::-1]
    top_3 = []
    for i in top_indices:
        cls_name = CLASS_NAMES[i]
        p_type, d_name, _ = parse_class_name(cls_name)
        top_3.append({
            "plant": p_type,
            "disease": d_name,
            "confidence": float(predictions[0][i])
        })
        
    return {
        "plant": plant_type,
        "disease": disease,
        "status": status,
        "confidence": confidence,
        "top_3": top_3
    }

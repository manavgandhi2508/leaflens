import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from utils.image_processing import load_image_from_upload, preprocess_image_for_model
from services.classifier import load_cnn_model, predict_disease
from services.symptom_analyzer import analyze_visual_symptoms
from services.recommendations import get_recommendations
from services.gemini_formatter import generate_formatted_report

st.set_page_config(
    page_title="LeafLens — AI Plant Leaf Health Analyzer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for polished UI
st.markdown("""
<style>
    .status-badge {
        display: inline-block;
        padding: 0.5em 1em;
        border-radius: 20px;
        font-weight: bold;
        color: white;
        margin: 10px 0;
    }
    .badge-healthy { background-color: #4CAF50; }
    .badge-attention { background-color: #FF9800; }
    .badge-disease { background-color: #f44336; }
    
    .footer-text {
        text-align: center;
        font-size: 0.8rem;
        color: #6c757d;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🌿 LeafLens")
    st.subheader("AI Powered Plant Health & Disease Analysis")
    st.write("Upload a clear photograph of a plant leaf to identify possible diseases, visible stress symptoms and recommended next steps.")
    st.markdown("---")

    # Load Model Early
    model = load_cnn_model()

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("📸 Image Input")
            uploaded_file = st.file_uploader(
                "Drag and drop a leaf image here", 
                type=['jpg', 'jpeg', 'png']
            )
            
            if uploaded_file is not None:
                image = load_image_from_upload(uploaded_file)
                st.image(image, use_container_width=True, caption="Uploaded Leaf Preview", output_format="PNG")
                st.caption(f"**Filename:** {uploaded_file.name} | **Resolution:** {image.size[0]}x{image.size[1]}px")
                analyze_button = st.button("🔍 Analyze Leaf", use_container_width=True, type="primary")
            else:
                analyze_button = False

    with col2:
        with st.container(border=True):
            st.subheader("🔬 Analysis Result")
            
            if uploaded_file is None:
                st.info("Upload a leaf image to begin analysis.")
            elif analyze_button:
                if model is None:
                    st.error("Error: Could not load the CNN model.")
                else:
                    with st.spinner("Analyzing plant health..."):
                        # 1. Image Preprocessing & CNN Inference
                        img_array = preprocess_image_for_model(image)
                        prediction_result = predict_disease(model, img_array)
                        
                        # 2. Visual Symptom Analysis
                        symptoms = analyze_visual_symptoms(image)
                        
                        # 3. Get Recommendations
                        recs = get_recommendations(
                            prediction_result["disease"], 
                            prediction_result["status"], 
                            symptoms
                        )
                        
                        # 4. Optional Gemini Language Format
                        formatted_report = generate_formatted_report(
                            prediction_result["plant"],
                            prediction_result["disease"],
                            prediction_result["confidence"],
                            symptoms
                        )

                    # -- UI Display --
                    st.markdown("### A. AI CLASSIFICATION")
                    st.write(f"**Plant:** {prediction_result['plant']}")
                    st.write(f"**Detected condition:** {prediction_result['disease']}")
                    st.write(f"**Model confidence:** {prediction_result['confidence']:.1%}")
                    
                    # Badge
                    if prediction_result["status"] == "Healthy":
                        st.markdown("<span class='status-badge badge-healthy'>🟢 HEALTHY</span>", unsafe_allow_html=True)
                    elif prediction_result["confidence"] < 0.6:
                        st.markdown("<span class='status-badge badge-attention'>🟠 ATTENTION REQUIRED</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span class='status-badge badge-disease'>🔴 DISEASE DETECTED</span>", unsafe_allow_html=True)
                    
                    with st.expander("Top 3 CNN Predictions"):
                        for i, t in enumerate(prediction_result["top_3"]):
                            st.write(f"{i+1}. {t['plant']} — {t['disease']}: {t['confidence']:.1%}")
                            
                    st.divider()
                    
                    st.markdown("### B. VISIBLE LEAF SYMPTOMS")
                    st.write("**Visible symptoms:**")
                    for obs in symptoms["observations"]:
                        st.write(f"• {obs}")
                        
                    st.divider()
                    
                    st.markdown("### C. COMPREHENSIVE REPORT (AI Formatted)")
                    st.markdown(formatted_report)

    # Technical Architecture Section
    st.markdown("---")
    st.header("How LeafLens Works")
    
    st.markdown("""
    ```mermaid
    graph TD;
        A[User Uploads Leaf Image] --> B[Image Preprocessing]
        B --> C[TensorFlow CNN]
        C --> D[38-Class Softmax Prediction]
        B --> E[Visual Symptom Analyzer]
        D --> F[Recommendation Engine]
        E --> F
        F --> G[Optional Gemini Formatter]
        G --> H[Leaf Health Report]
    ```
    """)
    
    st.markdown("""
    **1 — INPUT**: User uploads an RGB photograph of a plant leaf.  
    **2 — PREPROCESSING**: The image is converted to RGB, resized to 128 × 128 pixels and transformed into the numerical format expected by the trained neural network.  
    **3 — CNN MODEL**: LeafLens uses a TensorFlow/Keras Convolutional Neural Network. Convolution layers learn visual patterns such as colour changes, spots, lesions and leaf textures.  
    **4 — CLASSIFICATION**: The final neural-network layer produces probability scores for 38 supported plant health and disease classes. The class with the highest probability becomes the primary prediction.  
    **5 — SYMPTOM ANALYSIS**: Basic computer-vision measurements estimate visible characteristics such as green, yellow and brown regions. These observations supplement the CNN prediction.  
    **6 — RECOMMENDATION ENGINE**: The disease prediction and visible symptoms are mapped to concise possible causes and general plant-care recommendations.  
    **7 — OPTIONAL LANGUAGE AI**: If configured, Gemini converts the structured prediction into simple natural-language guidance. Gemini does not make the original disease prediction.
    """)

    st.markdown("---")
    col3, col4 = st.columns(2)
    
    with col3:
        with st.container(border=True):
            st.subheader("Model Details")
            st.write("**Model type:** Convolutional Neural Network (CNN)")
            st.write("**Framework:** TensorFlow / Keras")
            st.write("**Input:** 128 × 128 RGB leaf image")
            st.write("**Output:** 38 classification probabilities")
            st.write("**Dataset:** New Plant Diseases Dataset / PlantVillage-derived dataset")
            st.write("**Approximate dataset size:** ~87,000 RGB leaf images")
            st.write("**Supported crops:** 14 plant categories including Tomato, Potato, Apple, Corn, Grape, Pepper and others.")
            st.write("**Prediction:** Highest Softmax probability")
        
    with col4:
        with st.container(border=True):
            st.subheader("Tech Stack")
            st.write("**Frontend / Interface:** Streamlit")
            st.write("**Programming Language:** Python")
            st.write("**Machine Learning:** TensorFlow + Keras CNN")
            st.write("**Image Processing:** Pillow + NumPy + OpenCV")
            st.write("**Data Handling:** Pandas")
            st.write("**Optional Language Generation:** Google Gemini API")
            st.write("**Model File:** .keras")

    with st.expander("Technical Explanation"):
        st.write("**CNN**")
        st.write("A Convolutional Neural Network learns spatial image features through convolution filters. Early layers detect simple features such as edges and colour transitions while deeper layers combine them into more complex disease-related visual patterns.")
        
        st.write("**Softmax**")
        st.write("The output layer assigns a probability to every supported class. The highest probability is selected as the prediction.")
        
        st.write("**Image preprocessing**")
        st.write("The uploaded image must be transformed to the same dimensions and numerical format used when the model was trained.")
        
        st.write("**Inference**")
        st.write("Inference means using an already-trained model to make a prediction on a new image. LeafLens performs inference; it does not train the CNN every time a user uploads an image.")
        
        st.write("**Gemini**")
        st.write("Gemini is optional and is used only for natural-language presentation of structured results.")

    st.markdown("<p class='footer-text'>LeafLens is an educational prototype. Results are preliminary and should not replace professional agricultural testing.</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

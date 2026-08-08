# LeafLens

## What it does
LeafLens is an educational prototype demonstrating how computer vision can assist with preliminary plant-health inspection. It uses a Convolutional Neural Network (CNN) to predict plant diseases and basic computer vision to identify visual symptoms like yellowing or browning. Optionally, it uses the Google Gemini API to formulate natural language summaries.

## Architecture
- **Frontend**: Streamlit
- **Machine Learning**: TensorFlow / Keras (CNN)
- **Image Processing**: OpenCV, Pillow, NumPy
- **Language Generation**: Google Gemini API (Optional)

## ML Model
The project uses a pre-trained Keras CNN model trained on an ~87,000 image dataset (derived from PlantVillage) to classify 38 plant and disease categories. Input images are resized to 128x128. 

## Installation

```bash
# Create a virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Activate it (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Gemini configuration (Optional)

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Then set:
```
GEMINI_API_KEY=PASTE_YOUR_FREE_GEMINI_API_KEY_HERE
```
*Note: The application will continue to work deterministically if Gemini is unavailable or not configured.*

## Running the application
```bash
streamlit run app.py
```

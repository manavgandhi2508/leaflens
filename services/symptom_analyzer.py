import cv2
import numpy as np
from PIL import Image

def analyze_visual_symptoms(image: Image.Image) -> dict:
    """
    Performs basic color thresholding to estimate visual symptoms.
    This is an approximation and shouldn't be treated as a definitive lab test.
    """
    # Convert PIL Image to OpenCV format (BGR)
    img_array = np.array(image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Define color bounds in HSV
    # Green (Healthy parts)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    
    # Yellow (Chlorosis / Nutrient stress)
    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([30, 255, 255])
    
    # Brown/Dark (Necrosis / Dead tissue / Spots)
    lower_brown = np.array([10, 20, 20])
    upper_brown = np.array([20, 255, 200])

    # Create masks
    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_brown = cv2.inRange(img_hsv, lower_brown, upper_brown)

    # Calculate pixel counts
    total_pixels = img_hsv.shape[0] * img_hsv.shape[1]
    
    green_pixels = cv2.countNonZero(mask_green)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    brown_pixels = cv2.countNonZero(mask_brown)

    green_pct = (green_pixels / total_pixels) * 100
    yellow_pct = (yellow_pixels / total_pixels) * 100
    brown_pct = (brown_pixels / total_pixels) * 100

    observations = []
    
    if green_pct > 60:
        observations.append("Leaf remains predominantly green")
    elif green_pct < 30:
        observations.append("Significant loss of healthy green tissue")

    if yellow_pct > 15:
        observations.append("Moderate to severe yellowing detected")
    elif yellow_pct > 5:
        observations.append("Mild yellowing/chlorosis visible")

    if brown_pct > 10:
        observations.append("Significant brown/dark regions visible")
    elif brown_pct > 2:
        observations.append("Small brown/dark spots detected")

    if not observations:
        observations.append("No distinct discoloration detected")

    return {
        "green_percentage": round(green_pct, 1),
        "yellow_percentage": round(yellow_pct, 1),
        "brown_percentage": round(brown_pct, 1),
        "observations": observations
    }

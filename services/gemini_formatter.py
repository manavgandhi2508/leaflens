import os
from google import genai
from google.genai import types

def generate_formatted_report(plant: str, disease: str, confidence: float, symptoms: dict) -> str:
    """
    Calls the Gemini API to format the structured data into a polished summary.
    Falls back to a deterministic string if no API key is provided or if an error occurs.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "PASTE_YOUR_FREE_GEMINI_API_KEY_HERE":
        return _deterministic_fallback(plant, disease, confidence, symptoms)

    try:
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        You are formatting an educational plant-health analysis. 
        Do not change the ML prediction or confidence. 
        Do not make definitive nutrient diagnoses. 
        Convert the supplied structured observations into concise easy-to-understand language.
        Keep the complete output under roughly 120 words.
        
        DATA:
        Plant: {plant}
        CNN Prediction: {disease}
        CNN Confidence: {confidence:.1%}
        Yellow Percentage: {symptoms.get('yellow_percentage')}%
        Brown Percentage: {symptoms.get('brown_percentage')}%
        Visual Observations: {', '.join(symptoms.get('observations', []))}
        
        Format as:
        **Summary**
        ...
        
        **Possible causes**
        ...
        
        **Recommended next steps**
        ...
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return _deterministic_fallback(plant, disease, confidence, symptoms)

def _deterministic_fallback(plant: str, disease: str, confidence: float, symptoms: dict) -> str:
    """Fallback text if Gemini is unavailable."""
    obs = ", ".join(symptoms.get('observations', []))
    
    report = f"**Summary**\n"
    report += f"The AI analysis predicts {disease} for this {plant} with {confidence:.1%} confidence. "
    report += f"Visual symptoms observed include: {obs}.\n\n"
    
    report += "**Possible causes**\n"
    if disease.lower() != "no disease detected":
        report += f"• Primary prediction: {disease}\n"
    else:
        report += "• Plant appears generally healthy.\n"
        
    if symptoms.get('yellow_percentage', 0) > 5:
        report += "• Yellowing may be consistent with several causes including nutrient imbalance.\n"
        
    report += "\n**Recommended next steps**\n"
    report += "• Check soil moisture before watering again.\n"
    report += "• Ensure appropriate light and airflow.\n"
    if disease.lower() != "no disease detected":
        report += "• Inspect neighbouring leaves for similar symptoms.\n"
        
    return report

from data.disease_info import DISEASE_INFO

def get_recommendations(disease: str, status: str, visual_symptoms: dict) -> list:
    """
    Generates a list of recommended actions based on the CNN disease prediction
    and the visual symptom observations.
    """
    recs = []
    
    if status == "Healthy":
        recs.append("💧 Maintain current watering schedule, checking soil moisture first.")
        recs.append("☀️ Ensure plant continues to receive appropriate light and airflow.")
        
        # Check if symptoms indicate otherwise despite "Healthy" CNN prediction
        if visual_symptoms.get("yellow_percentage", 0) > 5:
            recs.append("🧪 Minor yellowing detected; consider checking soil pH or nutrients if it worsens.")
        return recs

    # Diseased
    recs.append("🌱 Inspect neighbouring leaves and plants for similar symptoms to prevent spread.")
    
    # Try to find specific treatment in our disease info
    # Re-map "disease" to the key used in DISEASE_INFO (e.g. "Early Blight" -> "Early_blight")
    mapped_key = disease.replace(" ", "_")
    disease_entry = None
    for k in DISEASE_INFO.keys():
        if k.lower() in mapped_key.lower():
            disease_entry = DISEASE_INFO[k]
            break
            
    if disease_entry and 'treatment' in disease_entry:
        recs.append(f"🛡️ Disease-specific: {disease_entry['treatment']}")
    else:
        recs.append("✂️ Consider removing heavily infected leaves if disease is strongly indicated.")
        
    # Symptom-based additions
    if visual_symptoms.get("yellow_percentage", 0) > 10:
        recs.append("🧪 Yellowing may be consistent with several causes including nutrient imbalance or water stress.")
    
    if visual_symptoms.get("brown_percentage", 0) > 5:
        recs.append("🍂 Dead or brown tissue will not recover; focus on protecting new growth.")
        
    return recs

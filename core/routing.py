"""
Southlake Medical AI Agent — Department Routing

Rule-based department suggestion with keyword matching.
AI can override via ai_client if available.
"""

from utils.constants import SYMPTOM_CATEGORIES


def detect_category(chief_complaint: str) -> str:
    """Detect symptom category from chief complaint keywords."""
    complaint = chief_complaint.lower()
    for cat_key, cat in SYMPTOM_CATEGORIES.items():
        for kw in cat["keywords"]:
            if kw.lower() in complaint:
                return cat_key
    return "general"


def suggest_department(chief_complaint: str, symptoms: dict = None) -> tuple[str, str]:
    """
    Suggest department based on complaint and symptoms.
    Returns (department, reason).
    """
    symptoms = symptoms or {}
    complaint = chief_complaint.lower()
    
    # Red-flag override → Emergency
    if symptoms.get("chest_pain") and symptoms.get("shortness_of_breath"):
        return "Emergency", "Chest pain with breathing difficulty requires emergency assessment"
    if symptoms.get("confusion") or symptoms.get("facial_droop") or symptoms.get("speech_difficulty"):
        return "Emergency", "Possible stroke symptoms — immediate neurological assessment needed"
    if symptoms.get("severe_bleeding"):
        return "Emergency", "Uncontrolled bleeding requires emergency intervention"
    if symptoms.get("loss_of_consciousness"):
        return "Emergency", "Loss of consciousness requires emergency evaluation"
    
    # Category-based routing
    category = detect_category(chief_complaint)
    if category in SYMPTOM_CATEGORIES:
        dept = SYMPTOM_CATEGORIES[category]["department"]
        label = SYMPTOM_CATEGORIES[category]["label"]
        return dept, f"Symptoms consistent with {label} presentation"
    
    return "Emergency", "Unable to categorize — routing to Emergency for assessment"

"""
Southlake Medical AI Agent — Follow-Up Classification

Classifies post-discharge follow-up answers into Green / Yellow / Red.
"""

from utils.constants import FOLLOWUP_GREEN_MAX, FOLLOWUP_YELLOW_MAX


def classify_followup(answers: dict) -> dict:
    """
    Classify follow-up questionnaire answers into risk level.
    Returns score, level, action, and explanation.
    """
    score = 0
    concerns = []

    # Pain assessment
    pain = answers.get("pain_level", 0)
    if pain >= 7:
        score += 20
        concerns.append(f"High pain level ({pain}/10)")
    elif pain >= 4:
        score += 10
        concerns.append(f"Moderate pain level ({pain}/10)")

    # Vital sign flags
    if answers.get("fever"):
        score += 15
        concerns.append("Fever reported")
    if answers.get("shortness_of_breath"):
        score += 25
        concerns.append("Shortness of breath")
    if answers.get("blood_sugar_high"):
        score += 15
        concerns.append("Elevated blood sugar")
    if answers.get("blood_pressure_abnormal"):
        score += 15
        concerns.append("Abnormal blood pressure")
    if answers.get("oxygen_low"):
        score += 25
        concerns.append("Low oxygen saturation")
    if answers.get("new_symptoms"):
        score += 15
        concerns.append("New symptoms since discharge")
    if answers.get("medication_noncompliant"):
        score += 10
        concerns.append("Medication non-compliance")
    if answers.get("wound_issue"):
        score += 15
        concerns.append("Wound complication reported")

    # Trend assessment
    if answers.get("worse_than_before"):
        score += 20
        concerns.append("Patient reports worsening condition")

    # Classification
    if score <= FOLLOWUP_GREEN_MAX:
        level = "Green"
        action = "No clinician intervention required. Case closed with audit trail."
        urgency = "Low — recovery on track"
    elif score <= FOLLOWUP_YELLOW_MAX:
        level = "Yellow"
        action = "Schedule administrative phone callback. Possible nurse follow-up or appointment."
        urgency = "Moderate — monitoring recommended"
    else:
        level = "Red"
        action = "Escalate: transport recommendation generated. Clinician review required."
        urgency = "High — deterioration or significant concern detected"

    return {
        "score": score,
        "level": level,
        "action": action,
        "urgency": urgency,
        "concerns": concerns,
    }

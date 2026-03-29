"""
Southlake Medical AI Agent — Triage Scoring Engine

Transparent, explainable priority scoring.
Weights: Age Risk 60% (0-60) + Pain 20% (0-20) + AI Severity 20% (0-20) = 0-100
Red-flag override forces minimum 85.
"""

from utils.constants import AGE_RISK_TABLE, PRIORITY_GREEN_MAX, PRIORITY_YELLOW_MAX


def calculate_age_risk(age: int) -> int:
    """Age-based risk score (0-60). Weight: 60%."""
    if age >= 85: return 60
    elif age >= 75: return 48
    elif age >= 65: return 36
    elif age >= 50: return 20
    elif age >= 18: return 12
    else: return 18  # pediatric


def calculate_pain_score(pain_level: int) -> int:
    """Pain score 0-20 from self-reported 0-10. Weight: 20%."""
    return min(max(pain_level, 0), 10) * 2


def calculate_ai_severity(symptoms: dict) -> int:
    """Rule-based severity from structured symptoms (0-20). Weight: 20%."""
    severity = 0
    weights = {
        "unable_to_walk": 7,
        "shortness_of_breath": 10,
        "rest_breathless": 10,
        "cannot_lie_flat": 7,
        "fever": 3,
        "swelling": 3,
        "mental_status_change": 14,
        "confusion": 14,
        "facial_droop": 14,
        "arm_weakness": 10,
        "speech_difficulty": 10,
        "chest_pain": 10,
        "diaphoresis": 7,
        "radiation_to_arm": 7,
        "severe_bleeding": 10,
        "rebound_tenderness": 7,
        "weight_loss": 3,
        "blood_in_stool": 7,
        "hematuria": 5,
    }
    for symptom, weight in weights.items():
        if symptoms.get(symptom):
            severity += weight
    return min(severity, 20)


def detect_red_flag(symptoms: dict) -> bool:
    """Check for red-flag symptoms that require immediate escalation."""
    red_flag_symptoms = [
        "shortness_of_breath",
        "rest_breathless",
        "chest_pain",
        "mental_status_change",
        "confusion",
        "facial_droop",
        "severe_bleeding",
        "loss_of_consciousness",
        "sudden_severe_headache",
    ]
    return any(symptoms.get(s) for s in red_flag_symptoms)


def get_priority_score(age: int, pain_level: int, symptoms: dict) -> dict:
    """
    Calculate full priority score with breakdown.
    Returns dict with all components and explanation.
    """
    pain_score = calculate_pain_score(pain_level)
    age_risk = calculate_age_risk(age)
    ai_severity = calculate_ai_severity(symptoms)
    red_flag = detect_red_flag(symptoms)

    total = pain_score + age_risk + ai_severity
    if red_flag:
        total = max(total, 85)

    if total <= PRIORITY_GREEN_MAX:
        level = "Green"
        band = "Low urgency — suitable for scheduled care"
    elif total <= PRIORITY_YELLOW_MAX:
        level = "Yellow"
        band = "Moderate urgency — should be seen within hours"
    else:
        level = "Red"
        band = "High urgency — requires immediate attention"

    # Build explanation
    explanations = []
    if pain_score >= 14:
        explanations.append(f"High pain level ({pain_level}/10)")
    if age_risk >= 36:
        explanations.append(f"Elevated age-related risk (age {age}, weight: 60%)")
    if ai_severity >= 10:
        explanations.append("Significant symptom severity detected")
    if red_flag:
        explanations.append(" RED FLAG symptoms present — automatic escalation")

    return {
        "pain_score": pain_score,
        "age_risk": age_risk,
        "ai_severity": ai_severity,
        "red_flag": red_flag,
        "total": total,
        "level": level,
        "band_description": band,
        "explanations": explanations,
    }

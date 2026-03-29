"""
Southlake Medical AI Agent — Transport Orchestration

Community shuttle node logic for Red-level follow-up patients.
"""

from utils.constants import TRANSPORT_NODES


def recommend_transport(level: str, has_family: bool = False, severe: bool = False) -> dict:
    """
    Recommend transport based on risk level and patient situation.
    
    Red Level 1: Family brings patient
    Red Level 2: Shuttle pickup from community node
    Red Level 3: Ambulance / emergency escalation
    """
    if level != "Red":
        return {
            "transport_level": "None",
            "recommendation": "No transport needed. Standard follow-up pathway.",
            "nodes": [],
            "icon": "",
        }

    if severe:
        return {
            "transport_level": "Red-LEVEL3",
            "recommendation": "URGENT: Immediate emergency escalation. Bypass community node — direct ambulance dispatch.",
            "nodes": [],
            "icon": "",
        }

    if has_family:
        return {
            "transport_level": "Red-LEVEL1",
            "recommendation": "Family member to bring patient to the nearest pickup node or directly to Southlake ED at a scheduled time.",
            "nodes": TRANSPORT_NODES,
            "icon": "",
        }

    return {
        "transport_level": "Red-LEVEL2",
        "recommendation": "Patient to proceed to designated community pickup node. Southlake shuttle will collect grouped patients on next available run.",
        "nodes": TRANSPORT_NODES,
        "icon": "",
    }

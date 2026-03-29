"""
Southlake Medical AI Agent — Doctor Queue Manager
"""


def build_queue_item(
    case_id: str,
    patient_id: str,
    patient_name: str,
    age: int,
    complaint: str,
    triage_result: dict,
    department: str,
) -> dict:
    """Build a queue entry from triage results."""
    return {
        "case_id": case_id,
        "patient_id": patient_id,
        "patient_name": patient_name,
        "age": age,
        "complaint_summary": complaint,
        "final_priority": triage_result["total"],
        "risk_level": triage_result["level"],
        "department": department,
        "red_flag": triage_result["red_flag"],
        "queue_status": "Waiting",
    }


def sort_queue(queue_list: list[dict]) -> list[dict]:
    """Sort queue by priority (highest first), then by red-flag status."""
    return sorted(
        queue_list,
        key=lambda x: (x.get("red_flag", False), x.get("final_priority", 0)),
        reverse=True,
    )

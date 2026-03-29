"""
Southlake — Synthetic Data Contract Builder
NL requirement → structured data contract
"""
import json
from utils.ai_client import gpt_generate_report


DEFAULT_CONTRACT = {
    "target_size": 500,
    "description": "",
    "geography": {"cities": [], "hospital_filter": None},
    "age_rules": {"min_age": None, "max_age": None, "elderly_pct_target": None},
    "department_mix": [],
    "diagnosis_filter": [],
    "escalation_target": None,
    "followup_targets": {"min_30d": None, "min_90d": None},
    "visit_channel_preference": None,
    "required_fields": [],
    "quality_thresholds": {"fidelity": 0.80, "utility": 0.70, "privacy_epsilon": 5.0},
    "privacy_thresholds": {"max_exact_match_rate": 0.05, "min_k_anonymity": 3},
}


CONTRACT_PROMPT_TEMPLATE = """Parse this healthcare synthetic data request into a structured JSON contract.

Request: "{prompt}"
Target records: {target_size}

Output ONLY valid JSON matching this schema exactly:
{{
  "target_size": {target_size},
  "description": "brief description of the request",
  "geography": {{
    "cities": ["list of cities or empty"],
    "hospital_filter": "hospital name substring or null"
  }},
  "age_rules": {{
    "min_age": integer or null,
    "max_age": integer or null,
    "elderly_pct_target": float 0-1 or null
  }},
  "department_mix": ["list of target departments or empty"],
  "diagnosis_filter": ["list of target diagnoses or empty"],
  "escalation_target": float 0-1 or null,
  "followup_targets": {{"min_30d": integer or null, "min_90d": integer or null}},
  "visit_channel_preference": "In-Person or Telephone or Hybrid or null",
  "required_fields": ["any special fields needed"],
  "quality_thresholds": {{"fidelity": 0.80, "utility": 0.70, "privacy_epsilon": 5.0}},
  "privacy_thresholds": {{"max_exact_match_rate": 0.05, "min_k_anonymity": 3}}
}}"""


def build_contract_from_prompt(prompt: str, target_size: int = 500) -> dict:
    """Use GPT to parse NL requirement into a data contract."""
    filled = CONTRACT_PROMPT_TEMPLATE.format(prompt=prompt, target_size=target_size)
    result = gpt_generate_report("Output only valid JSON. No markdown, no explanation.", filled)

    if result:
        try:
            s = result.find("{")
            e = result.rfind("}") + 1
            if s >= 0:
                parsed = json.loads(result[s:e])
                # Merge with defaults to ensure all fields exist
                contract = {**DEFAULT_CONTRACT, **parsed}
                contract["target_size"] = target_size
                return contract
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: keyword-based contract
    contract = DEFAULT_CONTRACT.copy()
    contract["target_size"] = target_size
    contract["description"] = prompt

    pl = prompt.lower()
    if "elderly" in pl or "senior" in pl or "65" in pl or "old" in pl:
        contract["age_rules"]["min_age"] = 65
    if "cardiac" in pl or "heart" in pl or "cardiology" in pl:
        contract["department_mix"] = ["Cardiology"]
        contract["diagnosis_filter"] = ["Congestive Heart Failure", "Stage II Hypertension"]
    if "mental" in pl:
        contract["department_mix"] = ["Mental Health"]
        contract["diagnosis_filter"] = ["Major Depressive Disorder"]
    if "respiratory" in pl or "copd" in pl or "asthma" in pl:
        contract["department_mix"] = ["Respiratory"]
        contract["diagnosis_filter"] = ["COPD", "Asthma"]
    if "oncology" in pl or "cancer" in pl:
        contract["department_mix"] = ["Oncology"]
        contract["diagnosis_filter"] = ["Breast Cancer", "Prostate Cancer"]
    if "southlake" in pl or "newmarket" in pl:
        contract["geography"]["hospital_filter"] = "Southlake"
    if "york" in pl:
        contract["geography"]["cities"] = ["Newmarket", "Aurora", "Richmond Hill", "Markham", "Vaughan"]

    # Extract escalation rate if mentioned
    import re
    esc_match = re.search(r'(\d+)%?\s*escalat', pl)
    if esc_match:
        contract["escalation_target"] = int(esc_match.group(1)) / 100

    return contract

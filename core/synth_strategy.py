"""
Southlake — Synthetic Data Strategy Selector
Agent decides which generation approach to use based on contract + seed data.
"""
import pandas as pd


def choose_generation_strategy(contract: dict, seed_df: pd.DataFrame) -> dict:
    """
    Agent-style strategy selection: analyzes contract constraints
    and seed data properties to pick the optimal generation method.
    Returns strategy name, reasoning, and constraint classification.
    """
    n = contract.get("target_size", 500)
    age_rules = contract.get("age_rules", {})
    dept_mix = contract.get("department_mix", [])
    diag_filter = contract.get("diagnosis_filter", [])
    esc_target = contract.get("escalation_target")
    geo = contract.get("geography", {})
    fu_targets = contract.get("followup_targets", {})

    # Count active constraints
    hard_constraints = []
    soft_constraints = []

    if esc_target is not None:
        hard_constraints.append(f"Escalation rate = {esc_target:.0%}")
    if age_rules.get("min_age"):
        hard_constraints.append(f"Age ≥ {age_rules['min_age']}")
    if age_rules.get("max_age"):
        hard_constraints.append(f"Age ≤ {age_rules['max_age']}")
    if dept_mix:
        hard_constraints.append(f"Department focus: {', '.join(dept_mix)}")
    if diag_filter:
        hard_constraints.append(f"Diagnosis filter: {', '.join(diag_filter[:3])}")
    if geo.get("hospital_filter"):
        hard_constraints.append(f"Hospital: {geo['hospital_filter']}")
    if geo.get("cities"):
        soft_constraints.append(f"City preference: {', '.join(geo['cities'][:3])}")
    if fu_targets.get("min_30d"):
        soft_constraints.append(f"Follow-up 30d ≥ {fu_targets['min_30d']}")

    # Check seed data availability
    seed_size = len(seed_df)
    ratio = n / max(seed_size, 1)

    # Check if longitudinal features are needed
    has_temporal = "initial_visit_date" in seed_df.columns and "last_visit_date" in seed_df.columns
    need_longitudinal = contract.get("need_longitudinal", False)

    # ─── Decision Logic ───
    reasons = []

    if need_longitudinal and has_temporal:
        strategy = "synthea_lifecycle_simulation"
        reasons.append("Longitudinal patient journeys requested → Synthea-style lifecycle simulation")
        reasons.append("Temporal fields available in seed data for trajectory modeling")
    elif len(hard_constraints) >= 4 or (n <= 2000 and len(hard_constraints) >= 2):
        strategy = "constraint_weighted_blender"
        reasons.append(f"High constraint count ({len(hard_constraints)} hard + {len(soft_constraints)} soft)")
        reasons.append("Constraint reweighting gives precise control over target distributions")
        if esc_target:
            reasons.append(f"Escalation rate target ({esc_target:.0%}) requires post-hoc adjustment")
    elif ratio > 3 and len(hard_constraints) <= 1:
        strategy = "distribution_matched_sampling"
        reasons.append(f"Large output ({n}) relative to seed ({seed_size}) with few constraints")
        reasons.append("Distribution matching preserves multivariate correlations at scale")
    elif len(hard_constraints) >= 2:
        strategy = "constraint_weighted_blender"
        reasons.append(f"Moderate constraints ({len(hard_constraints)} hard) favor weighted blending")
        reasons.append("Seed data sufficient for reweighted sampling without CTGAN overhead")
    else:
        strategy = "distribution_matched_sampling"
        reasons.append("Low constraint count — standard distribution-matched sampling is sufficient")
        reasons.append("Preserves seed data statistical properties with minimal intervention")

    # Risk notes
    risk_notes = []
    if ratio > 5:
        risk_notes.append(" High expansion ratio — synthetic duplicates likely. Consider CTGAN for large-scale generation.")
    if len(hard_constraints) == 0:
        risk_notes.append(" No hard constraints specified — output will mirror seed distribution closely.")
    if esc_target and esc_target > 0.5:
        risk_notes.append(" High escalation target may produce clinically implausible cohort.")

    return {
        "strategy_name": strategy,
        "display_name": {
            "constraint_weighted_blender": "Constraint-Weighted Cohort Blender",
            "distribution_matched_sampling": "Distribution-Matched Sampling",
            "synthea_lifecycle_simulation": "Synthea-Style Lifecycle Simulation",
        }.get(strategy, strategy),
        "reasons": reasons,
        "hard_constraints": hard_constraints,
        "soft_constraints": soft_constraints,
        "risk_notes": risk_notes,
        "seed_size": seed_size,
        "expansion_ratio": round(ratio, 2),
    }

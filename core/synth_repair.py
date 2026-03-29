"""
Southlake — Synthetic Data Repair Engine
D0 → Audit → Repair → D1 with before/after metrics.
"""
import pandas as pd
import numpy as np
import hashlib


def needs_repair(struct_audit: dict, plaus_audit: dict, priv_audit: dict, contract: dict) -> tuple[bool, list[str]]:
    """Determine if repair pass is needed. Returns (needs_repair, reasons)."""
    reasons = []

    if struct_audit.get("verdict") == "FAIL":
        reasons.append("Structural audit FAILED")
    if struct_audit.get("total_issues", 0) > 3:
        reasons.append(f"Structural issues: {struct_audit['total_issues']} issue types")

    if plaus_audit.get("fidelity_mean", 1) < contract.get("quality_thresholds", {}).get("fidelity", 0.8):
        reasons.append(f"Fidelity below threshold ({plaus_audit.get('fidelity_mean', 0):.2f} < {contract.get('quality_thresholds', {}).get('fidelity', 0.8)})")

    if plaus_audit.get("utility_pass_rate", 1) < contract.get("quality_thresholds", {}).get("utility", 0.7):
        reasons.append(f"Utility below threshold ({plaus_audit.get('utility_pass_rate', 0):.0%} < {contract.get('quality_thresholds', {}).get('utility', 0.7):.0%})")

    priv_thresh = contract.get("privacy_thresholds", {})
    if priv_audit.get("exact_match_rate", 0) > priv_thresh.get("max_exact_match_rate", 0.05):
        reasons.append(f"Exact match rate too high ({priv_audit.get('exact_match_rate', 0):.1%} > {priv_thresh.get('max_exact_match_rate', 0.05):.0%})")

    if priv_audit.get("k_anonymity_min", 999) < priv_thresh.get("min_k_anonymity", 3):
        reasons.append(f"k-anonymity too low (min k={priv_audit.get('k_anonymity_min', 0)} < {priv_thresh.get('min_k_anonymity', 3)})")

    if priv_audit.get("overall_risk") == "HIGH":
        reasons.append("Overall privacy risk rated HIGH")

    return len(reasons) > 0, reasons


def repair_dataset(df: pd.DataFrame, contract: dict, struct_audit: dict, priv_audit: dict, seed_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Repair pass: fix structural issues, reduce privacy risk, re-align constraints.
    Returns (repaired_df, repair_report).
    """
    repaired = df.copy()
    repairs = []
    metrics_before = {}
    metrics_after = {}

    # ─── 1. Fix temporal inconsistency ───
    if "initial_visit_date" in repaired.columns and "last_visit_date" in repaired.columns:
        init = pd.to_datetime(repaired["initial_visit_date"], errors="coerce")
        last = pd.to_datetime(repaired["last_visit_date"], errors="coerce")
        bad_mask = last < init
        n_bad = bad_mask.sum()
        metrics_before["temporal_inconsistency"] = int(n_bad)
        if n_bad > 0:
            # Swap dates where last < init
            repaired.loc[bad_mask, ["initial_visit_date", "last_visit_date"]] = repaired.loc[bad_mask, ["last_visit_date", "initial_visit_date"]].values
            repairs.append(f"Fixed {n_bad} temporal inconsistencies (swapped init/last dates)")
        metrics_after["temporal_inconsistency"] = 0

    # ─── 2. Fix followup count consistency ───
    if "followup_count_30d" in repaired.columns and "followup_count_90d" in repaired.columns:
        bad_fu = repaired["followup_count_30d"] > repaired["followup_count_90d"]
        n_bad = bad_fu.sum()
        metrics_before["followup_inconsistency"] = int(n_bad)
        if n_bad > 0:
            repaired.loc[bad_fu, "followup_count_90d"] = repaired.loc[bad_fu, "followup_count_30d"]
            repairs.append(f"Fixed {n_bad} followup count inconsistencies (30d > 90d)")
        metrics_after["followup_inconsistency"] = 0

    # ─── 3. Re-align escalation rate ───
    esc_target = contract.get("escalation_target")
    if esc_target is not None and "care_status" in repaired.columns:
        current_rate = (repaired["care_status"] == "Escalated").mean()
        metrics_before["escalation_rate"] = round(current_rate, 4)
        target = esc_target
        n_total = len(repaired)
        n_target = int(n_total * target)
        n_current = (repaired["care_status"] == "Escalated").sum()

        if abs(n_current - n_target) > n_total * 0.02:  # >2% off
            statuses = ["Active", "Monitoring", "Graduated"]
            if n_target > n_current:
                non_esc = repaired[repaired["care_status"] != "Escalated"].index
                to_flip = np.random.choice(non_esc, size=min(n_target - n_current, len(non_esc)), replace=False)
                repaired.loc[to_flip, "care_status"] = "Escalated"
            else:
                esc_idx = repaired[repaired["care_status"] == "Escalated"].index
                to_flip = np.random.choice(esc_idx, size=min(n_current - n_target, len(esc_idx)), replace=False)
                repaired.loc[to_flip, "care_status"] = np.random.choice(statuses, size=len(to_flip))

            new_rate = (repaired["care_status"] == "Escalated").mean()
            repairs.append(f"Re-aligned escalation rate: {current_rate:.1%} → {new_rate:.1%} (target: {target:.0%})")
            metrics_after["escalation_rate"] = round(new_rate, 4)
        else:
            metrics_after["escalation_rate"] = metrics_before["escalation_rate"]

    # ─── 4. Reduce exact match / privacy risk ───
    match_cols = [c for c in ["age", "city", "department", "primary_diagnosis", "care_status", "medication_plan"] if c in repaired.columns and c in seed_df.columns]
    if match_cols:
        # Hash synthetic rows
        syn_hashes = {}
        for idx, row in repaired[match_cols].iterrows():
            h = hashlib.md5(str(tuple(row.values)).encode()).hexdigest()
            syn_hashes[idx] = h

        seed_hash_set = set()
        for _, row in seed_df[match_cols].iterrows():
            h = hashlib.md5(str(tuple(row.values)).encode()).hexdigest()
            seed_hash_set.add(h)

        # Find duplicates
        dup_indices = [idx for idx, h in syn_hashes.items() if h in seed_hash_set]
        metrics_before["exact_matches"] = len(dup_indices)

        if len(dup_indices) > 0:
            # Perturb duplicates: shift age by 1-5, shuffle city
            for idx in dup_indices:
                repaired.at[idx, "age"] = max(18, min(95, repaired.at[idx, "age"] + np.random.choice([-3, -2, 2, 3])))
                if "city" in repaired.columns:
                    cities = seed_df["city"].unique().tolist()
                    repaired.at[idx, "city"] = np.random.choice(cities)

            repairs.append(f"Perturbed {len(dup_indices)} near-duplicate records to reduce privacy risk")

        # Recount
        new_syn_hashes = set()
        for _, row in repaired[match_cols].iterrows():
            h = hashlib.md5(str(tuple(row.values)).encode()).hexdigest()
            new_syn_hashes.add(h)
        new_matches = len(new_syn_hashes & seed_hash_set)
        metrics_after["exact_matches"] = new_matches

    # ─── 5. Fix invalid enum values ───
    enum_map = {
        "gender": ["Male", "Female", "Nonbinary"],
        "care_status": ["Active", "Monitoring", "Escalated", "Graduated"],
        "visit_channel": ["In-Person", "Telephone", "Hybrid"],
    }
    for col, valid in enum_map.items():
        if col in repaired.columns:
            invalid = ~repaired[col].isin(valid)
            n_inv = invalid.sum()
            if n_inv > 0:
                repaired.loc[invalid, col] = np.random.choice(valid, size=n_inv)
                repairs.append(f"Fixed {n_inv} invalid {col} values")

    report = {
        "repairs_applied": len(repairs),
        "repair_details": repairs,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
    }

    return repaired, report

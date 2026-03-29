"""
LakeProbe — User Override Store (Interactive Refinement Loop)
Functions:
Users can correct the system’s binding errors, and the corrected results are persisted as override rules.
In subsequent queries, Fusion Engine will prioritize existing overrides.

Design Highlights:
  1. Override granularity: hint → column (not query-level; can be generalized to synonymous queries)
  2. Overrides do not bypass hard constraints (incompatible data types are still blocked)
  3. Each override record includes usage count and source query, supporting ablation analysis
  4. Override boost is a bonus, not a requirement—if the column corrected by the user is indeed incompatible,
     the hard filter will still block it; this is “interactive but still evidence-grounded”
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Directory
OVERRIDE_DIR = Path(__file__).parent.parent / "data" / "overrides"
OVERRIDE_DIR.mkdir(parents=True, exist_ok=True)
OVERRIDE_FILE = OVERRIDE_DIR / "override_rules.json"


# Data Models
class OverrideRule(BaseModel):
    """A user-provided correction rule."""
    rule_id: str                         # Unique identifier
    hint: str                            # Semantic hint
    hint_type: str                       # "metric" | "dimension" | "time" | "filter"
    wrong_column: Optional[str] = None   # Previously selected column
    correct_column: str                  # User-corrected column
    dataset_id: str                      # Applicable dataset
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    source_query: str = ""               # Original query that triggered correction
    source_query_id: str = ""            # Associated audit record ID
    times_applied: int = 0               # Number of times reused
    last_applied_at: Optional[str] = None
    active: bool = True                  # Whether the rule is active


class OverrideResult(BaseModel):
    """Override application result, recorded in audit logs."""
    rules_checked: int = 0               # Number of rules checked
    rules_applied: int = 0               # Number of rules actually applied
    applied_details: list[dict] = Field(default_factory=list)


# Override Store (CRUD)
class OverrideStore:
    """
    Persistent storage for override rules.

    Uses both in-memory and JSON file storage to ensure persistence across restarts.
    """

    def __init__(self):
        self._rules: list[OverrideRule] = []
        self._load()

    def _load(self):
        """Load rules from disk."""
        if OVERRIDE_FILE.exists():
            try:
                data = json.loads(OVERRIDE_FILE.read_text(encoding="utf-8"))
                self._rules = [OverrideRule(**r) for r in data]
                logger.info(f"[Override] Loaded {len(self._rules)} rules")
            except Exception as e:
                logger.warning(f"[Override] Failed to load rules: {e}")
                self._rules = []

    def _save(self):
        """Persist rules to disk."""
        data = [r.model_dump() for r in self._rules]
        OVERRIDE_FILE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Create
    def add_rule(
        self,
        hint: str,
        hint_type: str,
        correct_column: str,
        dataset_id: str,
        wrong_column: str = None,
        source_query: str = "",
        source_query_id: str = "",
    ) -> OverrideRule:
        """
        Add a new override rule.

        If a rule with the same (hint, hint_type, dataset_id) exists,
        update the existing rule instead.
        """
        # Remove duplicates / Update
        existing = self._find_rule(hint, hint_type, dataset_id)
        if existing:
            existing.correct_column = correct_column
            existing.wrong_column = wrong_column
            existing.source_query = source_query
            existing.source_query_id = source_query_id
            existing.created_at = datetime.now().isoformat()
            existing.active = True
            self._save()
            logger.info(f"[Override] Updated rule: {hint} ({hint_type}) → {correct_column}")
            return existing

        rule_id = f"ovr_{len(self._rules)+1:04d}_{hint}_{hint_type}"
        rule = OverrideRule(
            rule_id=rule_id,
            hint=hint,
            hint_type=hint_type,
            wrong_column=wrong_column,
            correct_column=correct_column,
            dataset_id=dataset_id,
            source_query=source_query,
            source_query_id=source_query_id,
        )
        self._rules.append(rule)
        self._save()
        logger.info(f"[Override] Added rule: {hint} ({hint_type}) → {correct_column}")
        return rule

    #Read
    def _find_rule(self, hint: str, hint_type: str,
                    dataset_id: str) -> Optional[OverrideRule]:
        """Exact match lookup."""
        for r in self._rules:
            if (r.hint.lower() == hint.lower() and
                r.hint_type == hint_type and
                r.dataset_id == dataset_id and
                r.active):
                return r
        return None

    def lookup(self, hint: str, hint_type: str,
               dataset_id: str) -> Optional[OverrideRule]:
        """
        Find an applicable override rule.

        Matching logic:
          1. Exact match (hint, hint_type, dataset_id)
          2. Fuzzy match via alias (e.g., "revenue" matches "sales")
        """
        # Exact match
        exact = self._find_rule(hint, hint_type, dataset_id)
        if exact:
            return exact

        # Alias (Fuzzy Matching)
        from config import ALIAS_REVERSE, ALIAS_LEXICON
        canonical = ALIAS_REVERSE.get(hint.lower(), hint.lower())
        all_terms = {canonical} | set(ALIAS_LEXICON.get(canonical, []))

        for r in self._rules:
            if not r.active or r.dataset_id != dataset_id or r.hint_type != hint_type:
                continue
            r_canonical = ALIAS_REVERSE.get(r.hint.lower(), r.hint.lower())
            if r_canonical == canonical or r.hint.lower() in all_terms:
                return r

        return None

    def get_all_rules(self, active_only: bool = True) -> list[OverrideRule]:
        """Retrieve all rules."""
        if active_only:
            return [r for r in self._rules if r.active]
        return list(self._rules)

    #Update

    def record_application(self, rule: OverrideRule):
        """Record that a rule has been applied."""
        rule.times_applied += 1
        rule.last_applied_at = datetime.now().isoformat()
        self._save()

    #Delete

    def deactivate_rule(self, rule_id: str) -> bool:
        """Deactivate a rule (soft delete)"""
        for r in self._rules:
            if r.rule_id == rule_id:
                r.active = False
                self._save()
                logger.info(f"[Override] Deactivated rule: {rule_id}")
                return True
        return False

    def clear_all(self):
        """Remove all rules."""
        self._rules = []
        self._save()
        logger.info("[Override] Cleared all rules")


# Override Applicator (For Fusion Engine)

# Override boost score:
# Should be high enough to dominate normal retrieval scores,
# but must not bypass hard constraint filtering (which happens before boosting).
OVERRIDE_BOOST = 5.0


def apply_overrides_to_candidates(
    merged: dict,
    dataset_id: str,
    store: OverrideStore,
) -> tuple[dict, OverrideResult]:
    """
    Apply override boosting to candidate columns before binding selection.

    Mechanism:
      For each (hint, hint_type), look up override rules.
      If a matching rule exists, boost the score of the correct_column.

      If the correct_column is not in candidates (filtered out by hard constraints),
      do NOT force injection — maintain evidence-grounded behavior.

    Returns:
        (modified_candidates, override_result)
    """
    result = OverrideResult()

    for hint_type in ["metric", "dimension", "time", "filter"]:
        for hint, cands in merged.get(hint_type, {}).items():
            result.rules_checked += 1

            rule = store.lookup(hint, hint_type, dataset_id)
            if rule is None:
                continue

            # Find a matching rule and try applying it
            applied = False
            for c in cands:
                if c.column_name == rule.correct_column:
                    old_score = c.score
                    c.score += OVERRIDE_BOOST
                    c.evidence.append(
                        f"user_override: {rule.wrong_column or '?'} → {rule.correct_column} "
                        f"(rule={rule.rule_id}, applied {rule.times_applied}x before)"
                    )
                    store.record_application(rule)
                    applied = True

                    result.rules_applied += 1
                    result.applied_details.append({
                        "rule_id": rule.rule_id,
                        "hint": hint,
                        "hint_type": hint_type,
                        "correct_column": rule.correct_column,
                        "old_score": round(old_score, 4),
                        "new_score": round(c.score, 4),
                    })
                    logger.info(
                        f"[Override] Applied {rule.rule_id}: "
                        f"{hint}({hint_type}) → {rule.correct_column} "
                        f"(score {old_score:.3f} → {c.score:.3f})"
                    )
                    break

            if not applied:
                logger.info(
                    f"[Override] Rule {rule.rule_id} matched hint '{hint}' "
                    f"but column '{rule.correct_column}' not in candidates "
                    f"(may have been hard-filtered)"
                )

            # Re-sort after boost
            cands.sort(key=lambda x: x.score, reverse=True)

    return merged, result


# Convenience: From audit record create overrides
def create_override_from_correction(
    store: OverrideStore,
    query_id: str,
    hint: str,
    hint_type: str,
    wrong_column: str,
    correct_column: str,
    dataset_id: str,
    raw_query: str = "",
) -> OverrideRule:
    """
    Called when the user corrects a binding in the UI.

    The correction is stored so that future similar queries
    can automatically reuse the same mapping
    """
    return store.add_rule(
        hint=hint,
        hint_type=hint_type,
        correct_column=correct_column,
        wrong_column=wrong_column,
        dataset_id=dataset_id,
        source_query=raw_query,
        source_query_id=query_id,
    )


# Singleton Store
_store_instance: Optional[OverrideStore] = None


def get_override_store() -> OverrideStore:
    """Singleton"""
    global _store_instance
    if _store_instance is None:
        _store_instance = OverrideStore()
    return _store_instance

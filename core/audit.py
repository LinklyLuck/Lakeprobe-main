"""
LakeProbe — Audit Trail
Fuction:
Save a complete audit trail for each query:
  1 Original user query
  2 QueryIntent
  3 Candidate datasets/columns
  4 Intercepted erroneous candidates
  5 Final binds and evidence
  6 Execution plan and results
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from core.models import (
    AuditRecord,
    BindingResult,
    ExecutablePlan,
    QueryIntent,
    RetrievalResult,
)

# Audit Storage Directory
AUDIT_DIR = Path(__file__).parent.parent / "data" / "audit_logs"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def create_audit_record(
    raw_query: str,
    intent: QueryIntent,
    candidates: RetrievalResult,
    binding: BindingResult,
    plan: ExecutablePlan,
    execution_result: dict,
    user_override: dict | None = None,
) -> AuditRecord:
    #Construct a complete AuditRecord
    record = AuditRecord(
        raw_query=raw_query,
        query_intent=intent.model_dump(),
        candidates=candidates.model_dump(),
        blocked_candidates=binding.blocked_candidates,
        final_binding=binding.model_dump(),
        executable_plan=plan.model_dump(),
        execution_summary={
            "columns": execution_result.get("columns", []),
            "row_count": execution_result.get("row_count", 0),
            "sql": execution_result.get("sql", ""),
            "error": execution_result.get("error"),
        },
        user_override=user_override,
    )
    return record


def save_audit(record: AuditRecord) -> Path:
    #Persist AuditRecord to a JSON file.
    filename = f"{record.timestamp.replace(':', '-')}_{record.query_id}.json"
    path = AUDIT_DIR / filename
    path.write_text(
        record.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return path


def load_audit(query_id: str) -> AuditRecord | None:
    #Load AuditRecord by query_id.
    for f in AUDIT_DIR.glob("*.json"):
        if query_id in f.name:
            return AuditRecord.model_validate_json(f.read_text(encoding="utf-8"))
    return None


def list_audits(limit: int = 20) -> list[dict]:
    #List the most recent audit summaries.
    files = sorted(AUDIT_DIR.glob("*.json"), reverse=True)[:limit]
    summaries = []
    for f in files:
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
            summaries.append({
                "query_id": rec.get("query_id"),
                "timestamp": rec.get("timestamp"),
                "raw_query": rec.get("raw_query"),
                "dataset": rec.get("final_binding", {}).get("dataset_id"),
                "row_count": rec.get("execution_summary", {}).get("row_count"),
                "has_error": bool(rec.get("execution_summary", {}).get("error")),
            })
        except Exception:
            pass
    return summaries


# UI Data Display (for Streamlit / Frontend)
def format_audit_for_display(record: AuditRecord) -> dict:
    """
    Format the AuditRecord for front-end display.
    Three core components:
      1. LLM's initial prediction (QueryIntent)
      2. Intercepted incorrect candidates
      3. Final binding and evidence
    """
    intent = record.query_intent

    #  Block 1: LLM's Initial Guess
    llm_guess = {
        "intent_type": intent.get("intent_type"),
        "metric_hints": intent.get("metric_hints", []),
        "dimension_hints": intent.get("dimension_hints", []),
        "filter_hints": intent.get("filter_hints", []),
        "time_hints": intent.get("time_hints", []),
        "agg_func": intent.get("agg_func_hint"),
        "ambiguities": intent.get("ambiguities", []),
    }

    # Block 2: Intercepted Candidates
    blocked = record.blocked_candidates

    # Block 3: Final Binding
    final = record.final_binding
    bindings_display = {
        "dataset": final.get("dataset_id"),
        "metrics": [
            {"hint": b["hint"], "→ column": b["column"],
             "score": b["score"], "evidence": b["evidence"]}
            for b in final.get("metric_bindings", [])
        ],
        "dimensions": [
            {"hint": b["hint"], "→ column": b["column"],
             "score": b["score"], "evidence": b["evidence"]}
            for b in final.get("dimension_bindings", [])
        ],
        "time": [
            {"hint": b["hint"], "→ column": b["column"],
             "score": b["score"], "evidence": b["evidence"]}
            for b in final.get("time_bindings", [])
        ],
    }

    return {
        "query": record.raw_query,
        "section_1_llm_guess": llm_guess,
        "section_2_blocked": blocked,
        "section_3_final_binding": bindings_display,
        "execution": record.execution_summary,
        "plan": record.executable_plan,
    }

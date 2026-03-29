"""
LakeProbe — Main Entry
Functions:
Two run modes:
  1. FastAPI REST API  (uvicorn app.main:api)
  2. Streamlit Demo UI (streamlit run app/main.py)
  Query → parse_intent → retrieve_candidates → fuse_and_plan
        → optimize_plan (Lazy) → [user review/edit] → execute → audit
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.models import QueryIntent, RetrievalResult, BindingResult, ExecutablePlan, AuditRecord, PlanOp, OpType
from core.parta_parser import build_query_intent
from core.profiler import build_profile_card, profile_all_csvs, load_profile_card
from core.dataset_card import build_dataset_card, build_column_index, load_all_dataset_cards
from core.retriever import retrieve_candidates
from core.fusion_engine import fuse_and_plan
from core.executor import execute_plan
from core.audit import create_audit_record, save_audit, format_audit_for_display, list_audits
from core.plan_optimizer import (
    estimate_plan_cost,
    optimize_plan,
    start_token_tracking,
    get_token_usage,
    measure_text2sql_tokens,
    plan_to_display,
    edit_plan_step,
    get_feedback_cache,
)
from config import CSV_DIR


# Initialize: Profile all CSVs → DatasetCard → ColumnIndex
MAX_CSV_SIZE = 500 * 1024 * 1024

def initialize_data(csv_dir: str | Path | None = None):
    """Scan CSV directory, build ProfileCard / DatasetCard / ColumnIndex."""
    d = Path(csv_dir) if csv_dir else CSV_DIR
    if not d.exists():
        print(f"[WARN] CSV directory not found: {d}")
        return []

    csv_files = list(d.glob("**/*.csv"))
    if not csv_files:
        print(f"[WARN] No CSV files in {d}")
        return []

    # Skip oversized files
    csv_files = [f for f in csv_files if f.stat().st_size < MAX_CSV_SIZE]

    print(f"[INIT] Found {len(csv_files)} CSV files, profiling...")
    cards = []
    for csv_path in csv_files:
        try:
            profile = build_profile_card(str(csv_path))
            ds_card = build_dataset_card(profile)
            col_index = build_column_index(profile)
            cards.append(ds_card)
            print(f"  ✓ {profile.dataset_id}: {profile.row_count} rows, {profile.col_count} cols")
        except Exception as e:
            print(f"  ✗ {csv_path.name}: {e}")
            continue

    print(f"[INIT] Done. {len(cards)} datasets indexed.")

    # Build join discovery index
    try:
        from core.join_discovery import build_join_index_all
        n_cols = build_join_index_all()
        print(f"[INIT] Join index: {n_cols} columns sketched.")
    except Exception as e:
        print(f"[INIT] Join index skipped: {e}")

    return cards


# Discovery Mode: dataset + column recommendation
_DISCOVERY_KEYWORDS = [
    "i want", "find me", "find a", "show me", "give me", "get me",
    "i need", "looking for", "search for", "which dataset",
    "recommend", "suggest", "discover", "explore",
    "dataset for", "data for", "data about", "dataset about",
    "predict", "classify", "cluster",
]

def _is_discovery_query(query: str) -> bool:
    """Detect if the query is exploratory (dataset discovery) vs analytical."""
    q = query.lower().strip()
    # Discovery pattern: matches exploration keywords
    if any(kw in q for kw in _DISCOVERY_KEYWORDS):
        return True
    return False


def discovery_pipeline(raw_query: str) -> dict:
    """
    Schema-Based Dataset Discovery:
      1. LLM generates "Desired Schema" from user query
      2. IR matches desired columns against all indexed datasets
      3. Returns ranked datasets with column-level match evidence
    Returns: {mode, query, desired_schema, matched_datasets, token_usage}
    """
    from core.schema_generator import generate_desired_schema, match_schema_to_datasets
    from core.profiler import load_profile_card

    token_usage = start_token_tracking()

    # Step 1: Generate desired schema from query
    desired = generate_desired_schema(raw_query)

    # Step 2: IR match against all indexed datasets
    results = match_schema_to_datasets(desired, raw_query=raw_query)

    # Step 3: Enrich results with full column details
    matched_datasets = []
    for r in results:
        profile = load_profile_card(r.dataset_id)
        col_details = []
        if profile:
            for col in profile.columns:
                col_info = {
                    "name": col.name,
                    "dtype": col.dtype,
                    "role": col.inferred_role.value,
                    "n_unique": col.n_unique,
                    "missing_rate": col.missing_rate,
                    "sample_values": col.sample_values[:5],
                }
                if col.min_val is not None:
                    col_info["min"] = col.min_val
                if col.max_val is not None:
                    col_info["max"] = col.max_val
                if col.mean_val is not None:
                    col_info["mean"] = round(col.mean_val, 2)
                col_details.append(col_info)

        matched_datasets.append({
            "dataset_id": r.dataset_id,
            "score": r.overall_score,
            "coverage": r.coverage,
            "row_count": r.row_count,
            "domain": r.domain,
            "summary": r.summary,
            "column_matches": [m.model_dump() for m in r.matched_columns],
            "all_columns": col_details,
        })

    # Token measurement
    if matched_datasets:
        profile_top = load_profile_card(matched_datasets[0]["dataset_id"])
        measure_text2sql_tokens(raw_query, profile_top)

    token_data = token_usage.to_dict() if token_usage else {}

    return {
        "mode": "discovery",
        "query": raw_query,
        "desired_schema": desired.model_dump(),
        "matched_datasets": matched_datasets,
        "token_usage": token_data,
    }


# Core query pipeline (with Lazy Execution)
def parse_intent_only(raw_query: str) -> dict:
    """
    Phase 1: Parse intent ONLY — for user review before proceeding.
    Returns intent + available choices for ambiguities.
    """
    token_usage = start_token_tracking()
    intent = build_query_intent(raw_query)

    # Generate smart choices based on indexed datasets
    choices = _generate_intent_choices(intent)

    return {
        "intent": intent.model_dump(),
        "choices": choices,
        "token_usage": token_usage.to_dict() if token_usage else {},
    }


def _generate_intent_choices(intent) -> dict:
    """
    Generate interactive choices for ambiguous/missing slots.
    Based on what's actually available in indexed datasets.
    """
    all_cards = load_all_dataset_cards()

    # Collect all available measures, dimensions, time columns from indexed data
    available_measures = set()
    available_dimensions = set()
    available_time = set()
    for card in all_cards:
        available_measures.update(card.measure_columns)
        available_dimensions.update(card.dimension_columns)
        available_time.update(card.time_columns)

    choices = {}

    # Metric choices (if missing or might be wrong)
    if not intent.metric_hints:
        if available_measures:
            choices["metric_missing"] = {
                "question": "Which metric do you want to analyze?",
                "options": sorted(available_measures)[:8],
                "allow_custom": True,
            }
    elif intent.metric_hints:
        # Even if LLM picked one, show alternatives
        choices["metric_verify"] = {
            "question": f"LLM picked metric: {intent.metric_hints}. Correct?",
            "current": intent.metric_hints,
            "options": sorted(available_measures)[:8],
            "allow_custom": True,
        }

    # Dimension choices (if missing)
    if not intent.dimension_hints:
        if intent.intent_type.value in ("aggregate", "ranking", "comparison"):
            if available_dimensions:
                choices["dimension_missing"] = {
                    "question": "Group by which dimension?",
                    "options": sorted(available_dimensions)[:8],
                    "allow_custom": True,
                }
    elif intent.dimension_hints:
        choices["dimension_verify"] = {
            "question": f"LLM picked dimensions: {intent.dimension_hints}. Correct?",
            "current": intent.dimension_hints,
            "options": sorted(available_dimensions)[:8],
            "allow_custom": True,
        }

    # Agg function choices (if ambiguous)
    if intent.intent_type.value in ("aggregate", "ranking") and not intent.agg_func_hint:
        choices["agg_missing"] = {
            "question": "Which aggregation function?",
            "options": ["sum", "avg", "count", "min", "max", "median"],
            "allow_custom": False,
        }

    # Time choices (if trend query with no time)
    if intent.intent_type.value == "trend" and not intent.time_hints:
        if available_time:
            choices["time_missing"] = {
                "question": "Which time column for the trend?",
                "options": sorted(available_time)[:5],
                "allow_custom": True,
            }

    # Ambiguity-specific choices
    q = intent.raw_query.lower()
    if any(k in q for k in ["best", "best_cn", "optimal_cn", "top"]) and not intent.metric_hints:
        choices["best_ambiguous"] = {
            "question": '"Best" by which measure?',
            "options": sorted(available_measures)[:6] if available_measures else ["sales", "profit", "quantity"],
            "allow_custom": True,
        }

    if any(k in q for k in ["recent", "recent_cn", "lately"]) and not intent.time_hints:
        choices["recent_ambiguous"] = {
            "question": '"Recent" means what time range?',
            "options": ["last 7 days", "last 30 days", "last quarter", "last year"],
            "allow_custom": True,
        }

    return choices


def pipeline_from_intent(
    raw_query: str,
    intent_dict: dict,
    auto_execute: bool = False,
    phase1_token_usage: dict = None,
) -> dict:
    """
    Phase 2: Run retrieval → fusion → plan from a CONFIRMED intent.
    Called after user reviews/edits the intent.

    phase1_token_usage: token data from Phase 1 (parse_intent_only), merged into final output.
    """
    token_usage = start_token_tracking()

    # Reconstruct QueryIntent from user-confirmed dict
    intent = QueryIntent(**intent_dict)

    # Step 1.5: Domain Routing (soft prior for retrieval)
    from core.domain_router import route_domain
    from config import DOMAIN_ROUTING_MODE
    domain_result = route_domain(intent.model_dump(), mode=DOMAIN_ROUTING_MODE)

    # Step 2: Retrieve Candidates (with domain boost)
    candidates = retrieve_candidates(intent, domain_prior=domain_result.predicted_domain)

    # Step 3: Fuse and Plan
    if not candidates.dataset_candidates:
        return {
            "error": "No matching datasets found.",
            "intent": intent.model_dump(),
        }

    binding, plan, override_info = fuse_and_plan(intent, candidates)

    # Step 4: Optimize Plan
    profile = load_profile_card(plan.dataset_id)
    optimized_plan, rewrites = optimize_plan(plan, profile=profile)
    plan_cost = estimate_plan_cost(optimized_plan)

    # Step 4b: Token measurement
    measure_text2sql_tokens(raw_query, profile)
    token_data = token_usage.to_dict() if token_usage else {}

    # Merge Phase 1 token usage (LLM intent parsing tokens)
    if phase1_token_usage:
        p1_lp = phase1_token_usage.get("lakeprobe_total_tokens", 0)
        p1_in = phase1_token_usage.get("lakeprobe_input_tokens", 0)
        p1_out = phase1_token_usage.get("lakeprobe_output_tokens", 0)
        token_data["lakeprobe_total_tokens"] = token_data.get("lakeprobe_total_tokens", 0) + p1_lp
        token_data["lakeprobe_input_tokens"] = token_data.get("lakeprobe_input_tokens", 0) + p1_in
        token_data["lakeprobe_output_tokens"] = token_data.get("lakeprobe_output_tokens", 0) + p1_out
        # Recalculate saving ratio
        lp_total = token_data.get("lakeprobe_total_tokens", 0)
        t2s_total = token_data.get("text2sql_total_tokens", 0)
        if t2s_total > 0 and lp_total > 0:
            saving = (t2s_total - lp_total) / t2s_total
            token_data["token_saving_ratio"] = f"{saving:.0%}"

    output = {
        "intent": intent.model_dump(),
        "domain_routing": domain_result.to_dict(),
        "candidates": candidates.model_dump(),
        "binding": binding.model_dump(),
        "plan": plan.model_dump(),
        "optimized_plan": optimized_plan.model_dump(),
        "plan_cost": plan_cost.to_dict(),
        "plan_rewrites": rewrites,
        "token_usage": token_data,
        "override_info": override_info,
    }

    if auto_execute:
        from core.models import OpType
        result = execute_plan(optimized_plan)
        try:
            fc = get_feedback_cache()
            actual_rows = result.get("row_count", 0)
            for step in optimized_plan.steps:
                if step.op == OpType.FILTER:
                    fc.record(
                        optimized_plan.dataset_id,
                        step.params.get("column", ""),
                        step.params.get("op", "="),
                        step.params.get("value"),
                        estimated_sel=plan_cost.filter_selectivity,
                        actual_rows=actual_rows,
                        total_rows=plan_cost.scan_rows,
                    )
        except Exception as _exc:
            import logging; logging.getLogger("lakeprobe").debug(f"Non-critical error: {_exc}")
        record = create_audit_record(
            raw_query=raw_query, intent=intent, candidates=candidates,
            binding=binding, plan=optimized_plan, execution_result=result,
            user_override=override_info if override_info.get("rules_applied", 0) > 0 else None,
        )
        save_audit(record)
        output["result"] = result
        output["audit_display"] = format_audit_for_display(record)
        output["audit_id"] = record.query_id

    return output

def query_pipeline(raw_query: str, auto_execute: bool = True) -> dict:
    """
    End-to-end query processing (Lazy Execution aware).

    auto_execute=True  -> traditional mode, execute directly
    auto_execute=False -> lazy mode, return plan without executing, wait for user confirmation
    """
    # ── Token tracking start ──
    token_usage = start_token_tracking()

    # Step 1: Parse Intent
    intent = build_query_intent(raw_query)

    # Step 1.5: Domain Routing
    from core.domain_router import route_domain
    from config import DOMAIN_ROUTING_MODE
    domain_result = route_domain(intent.model_dump(), mode=DOMAIN_ROUTING_MODE)

    # Step 2: Retrieve Candidates (with domain boost)
    candidates = retrieve_candidates(intent, domain_prior=domain_result.predicted_domain)

    # Step 3: Fuse and Plan
    if not candidates.dataset_candidates:
        return {
            "error": "No matching datasets found. Please check your CSV data directory.",
            "intent": intent.model_dump(),
        }

    binding, plan, override_info = fuse_and_plan(intent, candidates)

    # Step 4: Optimize Plan (Lazy Execution)
    profile = load_profile_card(plan.dataset_id)
    optimized_plan, rewrites = optimize_plan(plan, profile=profile)
    plan_cost = estimate_plan_cost(optimized_plan)

    # Step 4b: Measure Text2SQL token cost via tiktoken (precise, not estimate)
    measure_text2sql_tokens(raw_query, profile)
    token_data = token_usage.to_dict() if token_usage else {}

    output = {
        "intent": intent.model_dump(),
        "domain_routing": domain_result.to_dict(),
        "candidates": candidates.model_dump(),
        "binding": binding.model_dump(),
        "plan": plan.model_dump(),
        "optimized_plan": optimized_plan.model_dump(),
        "plan_cost": plan_cost.to_dict(),
        "plan_rewrites": rewrites,
        "token_usage": token_data,
        "override_info": override_info,
    }

    # Step 5: Execute (or defer for Lazy mode)
    if auto_execute:
        result = execute_plan(optimized_plan)

        # Step 5b: Record feedback (actual rows vs estimate)
        try:
            fc = get_feedback_cache()
            actual_rows = result.get("row_count", 0)
            for step in optimized_plan.steps:
                if step.op == OpType.FILTER:
                    fc.record(
                        optimized_plan.dataset_id,
                        step.params.get("column", ""),
                        step.params.get("op", "="),
                        step.params.get("value"),
                        estimated_sel=plan_cost.filter_selectivity,
                        actual_rows=actual_rows,
                        total_rows=plan_cost.scan_rows,
                    )
        except Exception as _exc:
            import logging; logging.getLogger("lakeprobe").debug(f"Non-critical error: {_exc}")

        record = create_audit_record(
            raw_query=raw_query, intent=intent, candidates=candidates,
            binding=binding, plan=optimized_plan, execution_result=result,
            user_override=override_info if override_info.get("rules_applied", 0) > 0 else None,
        )
        save_audit(record)
        output["result"] = result
        output["audit_display"] = format_audit_for_display(record)
        output["audit_id"] = record.query_id

    return output


def execute_edited_plan(plan_dict: dict) -> dict:
    """Execute user-edited plan."""
    plan = ExecutablePlan(**plan_dict)
    return execute_plan(plan)


# Mode A: FastAPI REST API
def create_api():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel as PydBaseModel

    api = FastAPI(title="LakeProbe", version="0.3.0",
                  description="Evidence-based NL→CSV Query Engine "
                              "with Lazy Execution & Interactive Refinement")

    api.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    class QueryRequest(PydBaseModel):
        query: str
        auto_execute: bool = True

    class IntentRequest(PydBaseModel):
        query: str

    class OverrideRequest(PydBaseModel):
        query_id: str = ""
        hint: str
        hint_type: str
        wrong_column: str = ""
        correct_column: str
        dataset_id: str
        raw_query: str = ""

    @api.on_event("startup")
    async def startup():
        initialize_data()

    @api.post("/query")
    async def full_query(req: QueryRequest):
        return query_pipeline(req.query, auto_execute=req.auto_execute)

    @api.post("/parse_intent")
    async def parse_intent(req: IntentRequest):
        return build_query_intent(req.query).model_dump()

    @api.post("/execute_plan")
    async def execute(req: dict):
        return execute_plan(ExecutablePlan(**req))

    @api.post("/edit_plan")
    async def edit_plan(req: dict):
        """Edit a plan step (Lazy Execution interaction)."""
        plan = ExecutablePlan(**req["plan"])
        edited = edit_plan_step(
            plan, step_index=req.get("step_index", 0),
            action=req.get("action", "remove"),
            new_params=req.get("new_params"),
        )
        cost = estimate_plan_cost(edited)
        return {"plan": edited.model_dump(), "cost": cost.to_dict()}

    @api.post("/override")
    async def add_override(req: OverrideRequest):
        from core.override_store import get_override_store
        store = get_override_store()
        rule = store.add_rule(
            hint=req.hint, hint_type=req.hint_type,
            correct_column=req.correct_column, wrong_column=req.wrong_column,
            dataset_id=req.dataset_id, source_query=req.raw_query,
            source_query_id=req.query_id,
        )
        return {"status": "ok", "rule": rule.model_dump()}

    @api.get("/overrides")
    async def get_overrides():
        from core.override_store import get_override_store
        return [r.model_dump() for r in get_override_store().get_all_rules(active_only=True)]

    @api.delete("/override/{rule_id}")
    async def delete_override(rule_id: str):
        from core.override_store import get_override_store
        return {"status": "ok" if get_override_store().deactivate_rule(rule_id) else "not_found"}

    @api.post("/override/rerun")
    async def rerun_with_override(req: dict):
        from core.override_store import get_override_store
        get_override_store().add_rule(
            hint=req["hint"], hint_type=req["hint_type"],
            correct_column=req["correct_column"],
            wrong_column=req.get("wrong_column", ""),
            dataset_id=req["dataset_id"], source_query=req.get("query", ""),
        )
        return query_pipeline(req["query"])

    @api.get("/audits")
    async def get_audits():
        return list_audits()

    return api


api = create_api()


# Mode B: Streamlit Demo UI
def run_streamlit():
    import streamlit as st

    st.set_page_config(page_title="LakeProbe", page_icon="images/logo.png", layout="wide")

    # ── Global CSS (applies to ALL modes) ──
    st.markdown("""
    <style>
    .block-container {
        padding-top: 0.8rem; padding-bottom: 0.5rem;
        padding-left: 1rem; padding-right: 1rem;
        max-width: 100%;
    }
    /* All caption text — forced black */
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] span,
    [data-testid="stCaptionContainer"] code,
    [data-testid="stCaptionContainer"] strong,
    [data-testid="stCaptionContainer"] em,
    .st-emotion-cache-nahz7x,
    .st-emotion-cache-nahz7x p {
        font-size: 1rem !important;
        color: #000000 !important;
        opacity: 1 !important;
        line-height: 1.55 !important;
    }
    /* Metric cards */
    [data-testid="stMetric"] {
        background: #f9fafb; border: 1px solid #e5e7eb;
        border-radius: 6px; padding: 10px 14px;
    }
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: 700 !important; color: #111827 !important; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem !important; color: #6b7280 !important; text-transform: uppercase; }
    /* Buttons */
    .stButton > button[kind="primary"] {
        background-color: #1a56db !important; border: none !important;
        font-weight: 600 !important; font-size: 1rem !important; border-radius: 6px !important;
    }
    /* Expander / DataFrame / Dividers */
    [data-testid="stExpander"] { border: 1px solid #e5e7eb !important; border-radius: 6px !important; }
    [data-testid="stDataFrame"] { border: 1px solid #e5e7eb; border-radius: 6px; }
    hr { border-color: #e5e7eb !important; margin: 8px 0 !important; }
    .stAlert { padding: 8px 12px !important; font-size: 0.95rem !important; }
    /* Sidebar */
    [data-testid="stSidebar"] { background: #f9fafb !important; border-right: 1px solid #e5e7eb !important; }
    /* Text input — larger font + placeholder */
    [data-testid="stTextInput"] input {
        font-size: 1.2rem !important;
        padding: 12px 16px !important;
    }
    [data-testid="stTextInput"] input::placeholder {
        font-size: 1.1rem !important;
    }
    /* Column dividers */
    [data-testid="stHorizontalBlock"] > div:not(:last-child) {
        border-right: 1px solid #e5e7eb; padding-right: 16px !important;
    }
    [data-testid="stHorizontalBlock"] > div:not(:first-child) { padding-left: 16px !important; }
    </style>
    """, unsafe_allow_html=True)

    # Logo + Title
    import base64
    from pathlib import Path

    logo_path = Path(__file__).parent / "images" / "logo.png"

    if logo_path.exists():
        logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0px;margin-bottom:0px;margin-top:15px;">'
            f'<img src="data:image/png;base64,{logo_b64}" style="height:40px;">'
            f'<h1 style="margin:0;">LakeProbe</h1>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown('<h1 style="margin-bottom:0;">LakeProbe</h1>', unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.header("Configuration")

        csv_root = Path(CSV_DIR)
        subfolders = sorted([f.name for f in csv_root.iterdir() if f.is_dir()])

        if subfolders:
            selected = st.selectbox("Dataset folder", subfolders)
            csv_dir = str(csv_root / selected)
        else:
            csv_dir = st.text_input("CSV Directory", value=str(CSV_DIR))

        csv_count = len(list(Path(csv_dir).glob("**/*.csv")))
        st.caption(f"Found {csv_count} CSV files")

        if st.button("Index entire folder"):
            with st.spinner(f"Profiling {csv_count} CSVs..."):
                cards = initialize_data(csv_dir)
                st.success(f"Indexed {len(cards)} datasets")

        st.divider()
        st.header("Indexed Datasets")
        ds_cards = load_all_dataset_cards()
        if ds_cards:
            st.success(f"{len(ds_cards)} datasets indexed")
        else:
            st.info("No datasets indexed yet. Select a folder and click Index.")

        st.divider()
        if st.button("New Query", use_container_width=True, type="primary"):
            for key in list(st.session_state.keys()):
                if key not in ("csv_dir_input",):
                    del st.session_state[key]
            st.rerun()

        # Mode indicator (reads from previous run's session state)
        current_mode = st.session_state.get("_mode", "—")
        st.divider()
        st.markdown(f"**Pipeline Mode:** {current_mode}")

    # ── Main area: Query ──
    st.markdown('<p style="font-size:1.3rem;font-weight:600;margin-bottom:4px;">Ask a question about your data:</p>', unsafe_allow_html=True)
    query = st.text_input("query_input",
                          placeholder="e.g., 'I want wine dataset to predict quality' or 'average Weight by Breed'",
                          label_visibility="collapsed")
    search_clicked = st.button("Search", type="primary", use_container_width=True)

    # Detect mode early so sidebar can display it
    if query:
        _is_disc = _is_discovery_query(query)
        st.session_state["_mode"] = "Discovery" if _is_disc else "Analytical"

    if query and search_clicked or (query and st.session_state.get("_q") == query) or (query and st.session_state.get("_query_mode") in ("intent_review", "running", "query")):
        # ── Route: Discovery vs Query ──
        is_discovery = _is_discovery_query(query)

        # Store mode for sidebar
        st.session_state["_mode"] = "Discovery" if is_discovery else "Analytical"

        if is_discovery:
            # DISCOVERY MODE — Compact 3-column layout
            with st.spinner("Searching for matching datasets..."):
                disc = discovery_pipeline(query)

            schema = disc.get("desired_schema", {})
            matched = disc.get("matched_datasets", [])
            td = disc.get("token_usage", {})

            if not matched:
                st.warning("No matching datasets found.")
            else:
                # ── 3-column layout ──
                dc_left, dc_center, dc_right = st.columns([1, 1.5, 1])

                # ─── LEFT: Schema Understanding + Token Cost ───
                with dc_left:
                    card_title = lambda t: st.markdown(
                        f'<div style="font-size:1.05rem;font-weight:700;color:#1a56db;'
                        f'border-bottom:2.5px solid #1a56db;padding-bottom:5px;margin-bottom:10px;">{t}</div>',
                        unsafe_allow_html=True)

                    card_title("Schema Understanding")
                    st.caption(f"**Domain:** {schema.get('domain', '?')} · "
                               f"**Task:** {schema.get('task_type', '?')}")
                    st.caption(f"**Target:** {schema.get('target_description', '?')}")
                    cols_schema = schema.get("desired_columns", [])
                    if cols_schema:
                        for c in cols_schema[:8]:
                            st.caption(f"· `{c['name']}` ({c.get('role', '')}, {c.get('expected_dtype', '')})")
                    st.markdown("---")

                    card_title("Token Cost")
                    lp_tok = td.get("lakeprobe_total_tokens", 0)
                    t2s_tok = td.get("text2sql_total_tokens", 0)
                    if lp_tok > 0:
                        st.caption(f"**LP:** {lp_tok:,} · **T2S:** {t2s_tok:,}")
                        if t2s_tok > 0:
                            st.caption(f"**Saving:** {td.get('token_saving_ratio', '?')}")

                # ─── CENTER: Top dataset details + Download ───
                with dc_center:
                    card_title("Top Match")
                    top = matched[0]
                    st.caption(f"**{top['dataset_id']}** · {top['row_count']} rows · "
                               f"coverage: {top.get('coverage', 0):.0%} · score: {top.get('score', 0):.2f}")

                    # Column matches table
                    col_matches = top.get("column_matches", [])
                    if col_matches:
                        import pandas as pd
                        match_df = pd.DataFrame([{
                            "Desired": cm["desired"],
                            "Actual": cm["actual_column"],
                            "Score": f"{cm['match_score']:.2f}",
                        } for cm in col_matches])
                        st.dataframe(match_df, use_container_width=True, hide_index=True, height=180)

                    # Download button
                    st.markdown("---")
                    try:
                        from config import CSV_DIR as _csv_dir
                        import pandas as pd
                        csv_path = list(Path(_csv_dir).glob(f"**/{top['dataset_id']}*.csv"))
                        if csv_path:
                            dl_df = pd.read_csv(csv_path[0], nrows=5000)
                            csv_data = dl_df.to_csv(index=False)
                            st.download_button(
                                "Download dataset (CSV)",
                                data=csv_data,
                                file_name=f"{top['dataset_id']}.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )
                    except Exception:
                        pass

                # ─── RIGHT: Other matches (ranked list) ───
                with dc_right:
                    card_title(f"All Matches ({len(matched)})")
                    for idx, ds in enumerate(matched[:8]):
                        score = ds.get("score", 0)
                        cov = ds.get("coverage", 0)
                        filled = int(cov * 10)
                        bar = "█" * filled + "░" * (10 - filled)
                        is_top = " ←" if idx == 0 else ""
                        st.caption(f"**{idx+1}.** `{ds['dataset_id'][:25]}` "
                                   f"({ds['row_count']} rows) {cov:.0%} {bar}{is_top}")

                    # All columns of top dataset (collapsible)
                    st.markdown("---")
                    all_cols = top.get("all_columns", [])
                    if all_cols:
                        with st.expander(f"Columns in {top['dataset_id']} ({len(all_cols)})"):
                            import pandas as pd
                            col_df = pd.DataFrame([{
                                "Col": c["name"], "Type": c["dtype"],
                                "Role": c["role"], "Uniq": c["n_unique"],
                            } for c in all_cols])
                            st.dataframe(col_df, use_container_width=True, hide_index=True, height=200)

        else:
            # QUERY MODE — Two-Phase Lazy Pipeline
            # Phase 1: Parse Intent → User Review
            # Phase 2: Retrieve → Fuse → Plan → (Execute)

            # Phase 1: Parse intent (only if new query)
            if st.session_state.get("_current_query") != query:
                with st.spinner("Phase 1: Parsing intent..."):
                    parsed = parse_intent_only(query)
                st.session_state["_current_query"] = query
                st.session_state["_parsed_intent"] = parsed
                st.session_state["_confirmed_intent"] = None
                st.session_state["_query_output"] = None
                st.session_state["_query_mode"] = "intent_review"
                st.session_state.pop("last_result", None)

            mode = st.session_state.get("_query_mode", "intent_review")
            parsed = st.session_state.get("_parsed_intent", {})
            intent_data = parsed.get("intent", {})
            choices = parsed.get("choices", {})

            # ── Phase 1 UI: Intent Review ──
            if mode == "intent_review":
                st.subheader("Phase 1: Review LLM Intent (before retrieval)")
                st.caption("This is where hallucinations originate. Review and correct before proceeding.")

                # Row 1: Intent type + Agg + Sort + Limit
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    intent_types = ["aggregate", "trend", "ranking", "filter",
                                    "comparison", "distribution", "correlation", "lookup"]
                    cur_type = intent_data.get("intent_type", "aggregate")
                    idx = intent_types.index(cur_type) if cur_type in intent_types else 0
                    intent_data["intent_type"] = st.selectbox("Intent type", intent_types, index=idx, key="edit_intent_type")
                with c2:
                    agg_funcs = ["sum", "avg", "count", "min", "max", "median", "count_distinct", ""]
                    cur_agg = intent_data.get("agg_func_hint") or ""
                    idx_a = agg_funcs.index(cur_agg) if cur_agg in agg_funcs else len(agg_funcs) - 1
                    new_agg = st.selectbox("Agg function", agg_funcs, index=idx_a, key="edit_agg")
                    intent_data["agg_func_hint"] = new_agg if new_agg else None
                with c3:
                    sort_opts = ["desc", "asc", ""]
                    cur_sort = intent_data.get("sort_hint") or ""
                    idx_s = sort_opts.index(cur_sort) if cur_sort in sort_opts else 2
                    new_sort = st.selectbox("Sort", sort_opts, index=idx_s, key="edit_sort")
                    intent_data["sort_hint"] = new_sort if new_sort else None
                with c4:
                    cur_limit = intent_data.get("limit_hint") or 0
                    new_limit = st.number_input("Limit (0=none)", value=int(cur_limit),
                                                min_value=0, max_value=10000, key="edit_limit")
                    intent_data["limit_hint"] = new_limit if new_limit > 0 else None

                # Row 2: Metrics + Dimensions + Time — text input only
                m1, m2, m3 = st.columns(3)
                with m1:
                    cur_metrics = intent_data.get("metric_hints", [])
                    edited_metrics = st.text_input("Metrics (comma-separated)", value=", ".join(cur_metrics), key="edit_metrics")
                    intent_data["metric_hints"] = [m.strip() for m in edited_metrics.split(",") if m.strip()]
                with m2:
                    cur_dims = intent_data.get("dimension_hints", [])
                    edited_dims = st.text_input("Dimensions (comma-separated)", value=", ".join(cur_dims), key="edit_dims")
                    intent_data["dimension_hints"] = [d.strip() for d in edited_dims.split(",") if d.strip()]
                with m3:
                    cur_time = intent_data.get("time_hints", [])
                    edited_time = st.text_input("Time hints (comma-separated)", value=", ".join(cur_time), key="edit_time")
                    intent_data["time_hints"] = [t.strip() for t in edited_time.split(",") if t.strip()]

                # Confirm button
                if st.button("Confirm intent → Run retrieval & planning", type="primary"):
                    st.session_state["_confirmed_intent"] = intent_data
                    st.session_state["_query_mode"] = "running"
                    st.rerun()

            # ── Phase 2: Run pipeline from confirmed intent ──
            elif mode == "running":
                confirmed = st.session_state.get("_confirmed_intent")
                if confirmed:
                    # Get Phase 1 token usage to merge
                    p1_tokens = st.session_state.get("_parsed_intent", {}).get("token_usage", {})
                    with st.spinner("Phase 2: Retrieval → Fusion → Planning..."):
                        output = pipeline_from_intent(
                            query, confirmed,
                            auto_execute=False,
                            phase1_token_usage=p1_tokens,
                        )
                    st.session_state["_query_output"] = output
                    st.session_state["_query_mode"] = "query"
                    st.rerun()

            # Store for tabs rendering
            if st.session_state.get("_query_mode") == "query":
                pass  # will be picked up below

    # QUERY MODE — HORIZONTAL 3-COLUMN LAYOUT (one-screen)
    # Left: Intent + Retrieval + Binding
    # Center: Plan + Execute + Result
    # Right: Token Cost + Guard + Audit
    if query and not _is_discovery_query(query) and st.session_state.get("_query_mode") == "query":
        output = st.session_state.get("_query_output", {})
        if not output:
            return

        st.markdown("""
        <style>
        /* === Academic Tool Style (VLDB Demo) === */
        /* Clean white, thin borders, high readability, no decorative noise */

        .block-container {
            padding-top: 0.8rem; padding-bottom: 0.5rem;
            padding-left: 1rem; padding-right: 1rem;
            max-width: 100%;
        }

        /* Card section headers — neutral blue, clean underline */
        .card-hdr {
            font-size: 1.05rem; font-weight: 700;
            color: #1a56db; letter-spacing: 0.02em;
            border-bottom: 2.5px solid #1a56db;
            padding-bottom: 5px; margin-bottom: 10px;
        }

        /* All caption text — forced near-black */
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p,
        [data-testid="stCaptionContainer"] span,
        [data-testid="stCaptionContainer"] code,
        [data-testid="stCaptionContainer"] strong,
        [data-testid="stCaptionContainer"] em,
        .st-emotion-cache-nahz7x,
        .st-emotion-cache-nahz7x p {
            font-size: 1rem !important;
            color: #000000 !important;
            opacity: 1 !important;
            line-height: 1.55 !important;
        }

        /* Metric cards — bordered, clean */
        [data-testid="stMetric"] {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 10px 14px;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.4rem !important;
            font-weight: 700 !important;
            color: #111827 !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.85rem !important;
            color: #6b7280 !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        /* Buttons — professional blue */
        .stButton > button[kind="primary"] {
            background-color: #1a56db !important;
            border: none !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            border-radius: 6px !important;
        }

        /* Expander — clean */
        [data-testid="stExpander"] {
            border: 1px solid #e5e7eb !important;
            border-radius: 6px !important;
        }

        /* Dataframe — tight */
        [data-testid="stDataFrame"] {
            border: 1px solid #e5e7eb;
            border-radius: 6px;
        }

        /* Dividers — subtle */
        hr { border-color: #e5e7eb !important; margin: 8px 0 !important; }

        /* Warning / Success boxes — tighter */
        .stAlert { padding: 8px 12px !important; font-size: 0.95rem !important; }

        /* Code blocks — monospace, dark bg */
        code { font-size: 0.92rem !important; }

        /* Sidebar — clean */
        [data-testid="stSidebar"] {
            background: #f9fafb !important;
            border-right: 1px solid #e5e7eb !important;
        }

        /* Text input — larger font */
        [data-testid="stTextInput"] input {
            font-size: 1.1rem !important;
            padding: 10px 14px !important;
        }

        /* Column dividers — vertical lines between columns */
        [data-testid="stHorizontalBlock"] > div:not(:last-child) {
            border-right: 1px solid #e5e7eb;
            padding-right: 16px !important;
        }
        [data-testid="stHorizontalBlock"] > div:not(:first-child) {
            padding-left: 16px !important;
        }
        </style>
        """, unsafe_allow_html=True)

        def card_title(title):
            st.markdown(f'<div class="card-hdr">{title}</div>', unsafe_allow_html=True)

        binding = output.get("binding", {})
        candidates = output.get("candidates", {})
        plan_data = output.get("optimized_plan", output.get("plan", {}))
        plan_cost = output.get("plan_cost", {})
        td = output.get("token_usage", {})
        blocked_list = binding.get("blocked_candidates", [])

        col_plan, col_retrieval, col_intent, col_right = st.columns([1.3, 1, 1, 1])

        # COL 1: Plan + Execute (was col_c)

        # COL 1: Execution Plan
        with col_plan:
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric("Scan", f"{plan_cost.get('scan_rows', 0):,}")
            with mc2:
                st.metric("Result", f"{plan_cost.get('result_rows', 0):,}")
            with mc3:
                st.metric("Cost", f"{plan_cost.get('cost_score', 0):.0f}/100")
            rewrites = output.get("plan_rewrites", [])
            if rewrites:
                st.success(" · ".join(rewrites))

            if "edited_plan" not in st.session_state or st.session_state.get("last_query") != query:
                st.session_state["edited_plan"] = plan_data
                st.session_state["last_query"] = query
                st.session_state.pop("last_result", None)
            edited_steps = st.session_state["edited_plan"].get("steps", [])
            icons = {"scan": "▸", "filter": "▸", "derive_time": "▸", "groupby": "▸",
                     "aggregate": "▸", "sort": "▸", "limit": "▸", "select": "▸", "join": "▸"}
            all_ds = load_all_dataset_cards()
            ds_names = [dc.dataset_id for dc in all_ds]

            for i, step in enumerate(edited_steps):
                op = step["op"]
                op_str = str(op).lower().replace("optype.", "").strip()
                params = step.get("params", {})
                ic = icons.get(op_str, "▸")

                if op_str == "scan":
                    current_ds = params.get("dataset", "")
                    if ds_names:
                        idx_ds = ds_names.index(current_ds) if current_ds in ds_names else 0
                        new_ds = st.selectbox(f"{ic} SCAN", ds_names, index=idx_ds, key=f"scan_{i}_{query}")
                        if new_ds != current_ds:
                            params["dataset"] = new_ds
                            st.session_state["edited_plan"]["dataset_id"] = new_ds
                    proj = params.get("projected_columns", [])
                    if proj:
                        st.caption(f"  project: {', '.join(proj[:5])}")

                elif op_str == "filter":
                    fc1, fc2, fc3, fc4 = st.columns([3, 1, 3, 1])
                    with fc1:
                        params["column"] = st.text_input(f"{ic} Filter", value=params.get("column", ""), key=f"fc_{i}_{query}")
                    with fc2:
                        ops = ["=", ">", "<", ">=", "<=", "!=", "in"]
                        cur_op = params.get("op", "=")
                        idx_op = ops.index(cur_op) if cur_op in ops else 0
                        params["op"] = st.selectbox("Op", ops, index=idx_op, key=f"fo_{i}_{query}")
                    with fc3:
                        new_val = st.text_input("Val", value=str(params.get("value", "")), key=f"fv_{i}_{query}")
                        try:
                            params["value"] = int(new_val)
                        except ValueError:
                            try:
                                params["value"] = float(new_val)
                            except ValueError:
                                params["value"] = new_val
                    with fc4:
                        if st.button("X", key=f"rm_{i}_{query}"):
                            edited_steps.pop(i)
                            st.session_state["edited_plan"]["steps"] = edited_steps
                            st.rerun()

                elif op_str == "aggregate":
                    ac1, ac2 = st.columns(2)
                    with ac1:
                        funcs = ["sum", "avg", "count", "min", "max", "median", "count_distinct"]
                        cur_f = params.get("func", "sum")
                        idx_f = funcs.index(cur_f) if cur_f in funcs else 0
                        params["func"] = st.selectbox(f"{ic} AGG", funcs, index=idx_f, key=f"af_{i}_{query}")
                    with ac2:
                        params["metric"] = st.text_input("Metric", value=params.get("metric", ""), key=f"am_{i}_{query}")

                elif op_str == "groupby":
                    keys = params.get("keys", [params.get("column", "")])
                    new_keys = st.text_input(f"{ic} GROUP BY", value=", ".join(keys), key=f"gb_{i}_{query}")
                    params["keys"] = [k.strip() for k in new_keys.split(",") if k.strip()]

                elif op_str == "sort":
                    sc1, sc2 = st.columns(2)
                    with sc1:
                        params["column"] = st.text_input(f"{ic} SORT", value=params.get("column", ""), key=f"sc_{i}_{query}")
                    with sc2:
                        orders = ["desc", "asc"]
                        cur_o = params.get("order", "desc")
                        idx_o = orders.index(cur_o) if cur_o in orders else 0
                        params["order"] = st.selectbox("Order", orders, index=idx_o, key=f"so_{i}_{query}")

                elif op_str == "limit":
                    params["n"] = st.number_input(f"{ic} LIMIT", value=params.get("n", 10),
                                                   min_value=1, max_value=10000, key=f"ln_{i}_{query}")

                else:
                    st.caption(f"{ic} **{op}** {json.dumps(params, ensure_ascii=False)}")

            if st.button("▶ Execute", type="primary", use_container_width=True):
                with st.spinner("Executing..."):
                    exec_result = execute_edited_plan(st.session_state["edited_plan"])
                st.session_state["last_result"] = exec_result
                st.rerun()

        # COL 4: Token Cost + Result

        with col_retrieval:
            card_title("Retrieval")
            ds_cands = candidates.get("dataset_candidates", [])
            st.caption(f"Datasets: {', '.join(ds_cands[:3])}")
            for ctype, clabel in [("metric_candidates", "Metric"), ("dimension_candidates", "Dim"), ("time_candidates", "Time")]:
                entries = candidates.get(ctype, [])
                if entries:
                    items = " · ".join(f"`{c.get('column_name', '')}` ({c.get('score', 0):.2f})" for c in entries[:3])
                    st.caption(f"**{clabel}:** {items}")
            st.markdown("---")

            card_title("Hallucination Guard")
            st.caption(f"**{len(blocked_list)}** candidates blocked")
            for bl in blocked_list[:5]:
                st.caption(f"· `{bl.get('column', '')}` — {bl.get('reason', '')[:40]}")
            st.markdown("---")

            card_title("Audit")
            audit_result = st.session_state.get("last_result")
            if audit_result:
                try:
                    intent_obj = QueryIntent(**output["intent"])
                    cands_obj = RetrievalResult(**output["candidates"])
                    bind_obj = BindingResult(**output["binding"])
                    plan_obj = ExecutablePlan(**st.session_state["edited_plan"])
                    record = create_audit_record(
                        raw_query=query, intent=intent_obj, candidates=cands_obj,
                        binding=bind_obj, plan=plan_obj, execution_result=audit_result,
                    )
                    audit_d = format_audit_for_display(record)
                    st.caption(f"Query: {audit_d.get('query', '?')}")
                    ds_name = audit_d.get('section_3_final_binding', {}).get('dataset', '?')
                    st.caption(f"Dataset: {ds_name}")
                    row_count = audit_d.get('execution', {}).get('row_count', '?')
                    st.caption(f"Rows: {row_count}")
                    with st.expander("Full JSON"):
                        st.json(audit_d)
                except Exception as e:
                    st.caption(f"Error: {e}")
            else:
                st.caption("Execute first.")

        # COL 3: Intent + Binding (was col_a)

        # COL 3: Intent + Binding
        with col_intent:
            intent_c = output.get("intent", {})
            st.caption(f"**{intent_c.get('intent_type', '?')}** · metric: {', '.join(intent_c.get('metric_hints', []))} · dim: {', '.join(intent_c.get('dimension_hints', []))}")
            st.caption(f"agg: {intent_c.get('agg_func_hint', '—')} · sort: {intent_c.get('sort_hint', '—')} · limit: {intent_c.get('limit_hint', '—')}")
            original_intent = st.session_state.get("_parsed_intent", {}).get("intent", {})
            diffs = [k for k in ["intent_type", "metric_hints", "dimension_hints", "agg_func_hint"]
                     if str(original_intent.get(k)) != str(intent_c.get(k))]
            if diffs:
                st.warning(f"Corrected: {', '.join(diffs)}")
            dr = output.get("domain_routing", {})
            if dr and dr.get("predicted_domain"):
                st.caption(f"Domain: **{dr['predicted_domain']}** ({dr.get('confidence', 0):.0%})")
            st.markdown("---")

            card_title("Binding")
            blocked_list = binding.get("blocked_candidates", [])
            if blocked_list:
                st.caption(f"{len(blocked_list)} blocked")
            for bt in ["metric_bindings", "dimension_bindings", "time_bindings", "filter_bindings"]:
                for b in binding.get(bt, []):
                    lbl = bt.replace("_bindings", "")[:3].upper()
                    zone = b.get("zone", "accept")
                    zicon = {"accept": "[OK]", "uncertain": "[?]", "reject": "[X]"}.get(zone, "·")
                    st.caption(f"{zicon} {lbl}: `{b.get('hint', '')}` → **{b.get('column', '')}** ({b.get('score', 0):.2f})")

        # COL 2: Retrieval + Guard + Audit

        with col_right:
            card_title("Token Cost")
            lp = td.get("lakeprobe_total_tokens", 0)
            t2s = td.get("text2sql_total_tokens", 0)
            if lp > 0:
                st.caption(f"**LP:** {lp:,}  **T2S:** {t2s:,}")
                if t2s > 0:
                    st.caption(f"**Saving: {td.get('token_saving_ratio', '?')}**")
                    mx = max(lp, t2s)
                    st.caption(f"LP {'█' * int(lp / mx * 10)}{'░' * (10 - int(lp / mx * 10))} {lp:,}")
                    st.caption(f"T2S {'█' * int(t2s / mx * 10)}{'░' * (10 - int(t2s / mx * 10))} {t2s:,}")
                p1_td = st.session_state.get("_parsed_intent", {}).get("token_usage", {})
                p1_tok = p1_td.get("lakeprobe_total_tokens", 0)
                if p1_tok:
                    st.caption(f"LLM: 1 call (Phase1: {p1_tok:,})")
                    st.caption(f"Phase2: IR-only, no LLM")
            else:
                st.caption("No token data.")
            st.markdown("---")

            card_title("Result")
            disp_result = st.session_state.get("last_result")
            if disp_result:
                if disp_result.get("error"):
                    st.error(disp_result["error"][:60])
                if disp_result.get("rows"):
                    import pandas as pd
                    df = pd.DataFrame(disp_result["rows"], columns=disp_result["columns"])
                    st.dataframe(df, use_container_width=True, height=200)
                    st.caption(f"{disp_result['row_count']} rows")
                    # Download button
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        "Download result (CSV)",
                        data=csv_data,
                        file_name="lakeprobe_result.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                if disp_result.get("sql"):
                    with st.expander("SQL"):
                        st.code(disp_result["sql"], language="sql")
            else:
                st.caption("Click Execute to run.")

# CLI
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        import uvicorn
        initialize_data()
        uvicorn.run(api, host="0.0.0.0", port=8000)
    elif len(sys.argv) > 1 and sys.argv[1] == "--cli":
        initialize_data()
        print("\n🔍 LakeProbe CLI — Enter queries (type 'quit' to exit)\n")
        while True:
            q = input("Query> ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            if not q:
                continue
            output = query_pipeline(q)
            if "error" in output and not output.get("result"):
                print(f"[ERROR] {output['error']}\n")
                continue

            cost = output.get("plan_cost", {})
            print(f"\n[COST] scan={cost.get('scan_rows', '?')}, "
                  f"result≈{cost.get('result_rows', '?')} rows, "
                  f"score={cost.get('cost_score', '?')}/100")
            for w in cost.get("warnings", []):
                print(f"  ⚠ {w}")

            tokens = output.get("token_usage", {})
            if tokens.get("lakeprobe_total_tokens", 0) > 0:
                print(f"[TOKENS] LakeProbe={tokens['lakeprobe_total_tokens']} "
                      f"vs Text2SQL≈{tokens['text2sql_estimated_tokens']} "
                      f"(saving {tokens['token_saving_ratio']})")

            result = output.get("result", {})
            if result.get("rows"):
                cols = result["columns"]
                print(f"\n{'  |  '.join(cols)}")
                print("-" * 60)
                for row in result["rows"][:20]:
                    print("  |  ".join(str(v) for v in row))
                print(f"\n({result['row_count']} rows)")
            if result.get("sql"):
                print(f"\nSQL: {result['sql']}")
            if result.get("error"):
                print(f"\n[EXEC ERROR] {result['error']}")
            print()
    else:
        run_streamlit()
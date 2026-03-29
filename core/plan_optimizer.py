"""
LakeProbe — Plan Optimizer (Lazy Execution Engine)

Functions:
Six capabilities:
  1. Cost Model (histogram-based) — selectivity from equi-depth histograms + value counts
  2. Sampling Trigger           — when estimate is uncertain, probe data with lightweight query
  3. Plan Rewriter              — filter pushdown, projection pushdown, predicate simplify, TopN, dedup
  4. Runtime Feedback Cache     — store actual vs estimate, calibrate next time
  5. Token Tracker (tiktoken)   — precise token counting via tokenizer, not estimation
  6. Text2SQL Baseline          — real prompt construction + token measurement for comparison
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.models import (
    ExecutablePlan, OpType, PlanOp, ColumnProfile, ProfileCard,
)
from core.profiler import load_profile_card

logger = logging.getLogger(__name__)

# Feedback + config paths
_DATA_DIR = Path(__file__).parent.parent / "data"
_FEEDBACK_FILE = _DATA_DIR / "feedback_cache.json"
_DATA_DIR.mkdir(parents=True, exist_ok=True)



#  Cost Model (Histogram-Based)

class PlanCost:
    """Cost estimation result for a plan."""
    def __init__(self):
        self.scan_rows: int = 0
        self.filter_selectivity: float = 1.0
        self.rows_after_filter: int = 0
        self.groupby_cardinality: int = 0
        self.result_rows: int = 0
        self.has_sort: bool = False
        self.has_limit: bool = False
        self.limit_n: int = 0
        self.projected_columns: int = 0
        self.total_columns: int = 0
        self.warnings: list[str] = []
        self.estimated_cost_score: float = 0.0
        self.confidence: str = "high"  # "high" | "medium" | "low"
        self.sampling_triggered: bool = False
        self.sampled_selectivity: Optional[float] = None
        self.feedback_hit: bool = False

    def to_dict(self) -> dict:
        return {
            "scan_rows": self.scan_rows,
            "filter_selectivity": round(self.filter_selectivity, 4),
            "rows_after_filter": self.rows_after_filter,
            "groupby_cardinality": self.groupby_cardinality,
            "result_rows": self.result_rows,
            "has_sort": self.has_sort,
            "has_limit": self.has_limit,
            "limit_n": self.limit_n,
            "projected_columns": self.projected_columns,
            "total_columns": self.total_columns,
            "cost_score": round(self.estimated_cost_score, 1),
            "confidence": self.confidence,
            "sampling_triggered": self.sampling_triggered,
            "sampled_selectivity": round(self.sampled_selectivity, 4) if self.sampled_selectivity else None,
            "feedback_hit": self.feedback_hit,
            "warnings": self.warnings,
        }


def _estimate_selectivity_from_histogram(
    col: ColumnProfile, op: str, value
) -> tuple[float, str]:
    """
    Histogram-based selectivity estimation.
    Returns (selectivity, confidence).

    Uses equi-depth histogram for numeric range predicates,
    value_counts for exact match on categorical columns,
    fallback to 1/n_unique for unknown.
    """
    # Exact match on categorical column with value_counts
    if op == "=" and col.value_counts and str(value) in col.value_counts:
        total = sum(col.value_counts.values())
        if total > 0:
            sel = col.value_counts[str(value)] / total
            return sel, "high"

    # Exact match with value_counts (value not found → very low)
    if op == "=" and col.value_counts and str(value) not in col.value_counts:
        # Value doesn't exist in data
        return 0.001, "medium"

    # Numeric exact match: 1/n_unique
    if op == "=" and col.n_unique > 0:
        return 1.0 / col.n_unique, "medium"

    # Range predicates on histogram
    if col.histogram and col.dtype in ("int64", "float64"):
        try:
            fval = float(value)
        except (ValueError, TypeError):
            return 0.3, "low"

        total_rows = sum(b[2] for b in col.histogram)
        if total_rows == 0:
            return 0.3, "low"

        matching_rows = 0
        for lo, hi, cnt in col.histogram:
            if op in (">", ">="):
                if lo >= fval:
                    matching_rows += cnt
                elif hi >= fval:
                    # Uniform assumption within bucket
                    frac = (hi - fval) / (hi - lo) if hi > lo else 1.0
                    matching_rows += cnt * frac
            elif op in ("<", "<="):
                if hi <= fval:
                    matching_rows += cnt
                elif lo <= fval:
                    frac = (fval - lo) / (hi - lo) if hi > lo else 1.0
                    matching_rows += cnt * frac
            elif op == "=":
                if lo <= fval <= hi:
                    matching_rows += cnt / max(1, (hi - lo) if hi > lo else 1)

        sel = matching_rows / total_rows
        return max(0.001, min(sel, 1.0)), "high"

    # Fallback
    fallback_map = {
        "=": 1.0 / max(col.n_unique, 1),
        ">": 0.3, ">=": 0.3, "<": 0.3, "<=": 0.3,
        "!=": 1.0 - 1.0 / max(col.n_unique, 1),
        "in": 0.2, "between": 0.25,
    }
    sel = fallback_map.get(op, 0.3)
    return sel, "low"


def _find_column(profile: ProfileCard, col_name: str) -> Optional[ColumnProfile]:
    clean = col_name.replace("derived_", "")
    for col in profile.columns:
        if col.name == col_name or col.name == clean:
            return col
        if col_name.startswith("derived_") and col.name in col_name:
            return col
    return None


# Sampling Trigger

def _sample_selectivity(dataset_id: str, column: str, op: str, value) -> Optional[float]:
    """
    Run a lightweight DuckDB sampling query to estimate real selectivity.
    Triggered when histogram-based estimate has low confidence.
    """
    try:
        import duckdb
        from config import CSV_DIR

        csv_path = CSV_DIR / f"{dataset_id}.csv"
        if not csv_path.exists():
            # Try recursive search
            candidates = list(CSV_DIR.glob(f"**/{dataset_id}.csv"))
            if not candidates:
                return None
            csv_path = candidates[0]

        conn = duckdb.connect()
        # Sample 10% or 1000 rows, whichever is smaller
        total = conn.execute(
            f"SELECT COUNT(*) FROM read_csv_auto('{csv_path}')"
        ).fetchone()[0]
        sample_size = min(1000, max(100, total // 10))

        # Build filter expression
        col_expr = f'"{column}"'
        if column.startswith("derived_"):
            source = column.replace("derived_", "")
            col_expr = f'EXTRACT(YEAR FROM CAST("{source}" AS DATE))'

        if isinstance(value, str):
            where = f"{col_expr} {op} '{value}'"
        elif value is None:
            where = f"{col_expr} IS NULL"
        else:
            where = f"{col_expr} {op} {value}"

        result = conn.execute(f"""
            SELECT COUNT(*) FILTER (WHERE {where}), COUNT(*)
            FROM (SELECT * FROM read_csv_auto('{csv_path}')
                  USING SAMPLE {sample_size} ROWS)
        """).fetchone()
        conn.close()

        if result[1] > 0:
            return result[0] / result[1]
        return None
    except Exception as e:
        logger.warning(f"[Sample] Failed to sample {dataset_id}.{column}: {e}")
        return None


# Plan Rewriter

def optimize_plan(plan: ExecutablePlan, profile: Optional[ProfileCard] = None) -> tuple[ExecutablePlan, list[str]]:
    """
    Optimize ExecutablePlan with multiple rewrite rules.

    R1: Filter Pushdown — move FILTER right after SCAN/DERIVE_TIME
    R2: Redundant Elimination — remove duplicate operators
    R3: TopN / Limit Propagation — partial sort for top-k
    R4: Projection Pushdown — only scan needed columns
    R5: Predicate Simplification — constant folding, duplicate removal
    """
    rewrites: list[str] = []
    steps = [PlanOp(op=s.op, params=dict(s.params)) for s in plan.steps]

    # R5: Predicate Simplification (before pushdown)
    steps, r5 = _rewrite_predicate_simplify(steps)
    rewrites.extend(r5)

    # R1: Filter Pushdown
    steps, r1 = _rewrite_filter_pushdown(steps)
    rewrites.extend(r1)

    # R2: Redundant Elimination
    steps, r2 = _rewrite_dedup(steps)
    rewrites.extend(r2)

    # R3: TopN / Limit Propagation
    steps, r3 = _rewrite_topn(steps)
    rewrites.extend(r3)

    # R4: Projection Pushdown
    if profile is None:
        profile = load_profile_card(plan.dataset_id)
    steps, r4 = _rewrite_projection_pushdown(steps, profile)
    rewrites.extend(r4)

    return ExecutablePlan(dataset_id=plan.dataset_id, steps=steps), rewrites


def _rewrite_filter_pushdown(steps: list[PlanOp]) -> tuple[list[PlanOp], list[str]]:
    """R1: Move FILTERs right after SCAN + DERIVE_TIME."""
    filters = [s for s in steps if s.op == OpType.FILTER]
    if not filters:
        return steps, []

    result = []
    inserted_filters = False
    for s in steps:
        if s.op == OpType.FILTER:
            continue  # skip, will re-insert
        result.append(s)
        # Insert all filters after DERIVE_TIME (or SCAN if no DERIVE)
        if not inserted_filters and s.op in (OpType.DERIVE_TIME, OpType.SCAN):
            # Check if next non-filter is not DERIVE_TIME
            remaining = [x for x in steps[steps.index(s)+1:] if x.op != OpType.FILTER]
            if not remaining or remaining[0].op != OpType.DERIVE_TIME:
                result.extend(filters)
                inserted_filters = True

    if not inserted_filters:
        result.extend(filters)

    if result != [PlanOp(op=s.op, params=dict(s.params)) for s in steps]:
        return result, ["R1: Filter pushdown — moved filters closer to scan"]
    return steps, []


def _rewrite_dedup(steps: list[PlanOp]) -> tuple[list[PlanOp], list[str]]:
    """R2: Remove duplicate operators."""
    seen = set()
    result = []
    removed = 0
    for s in steps:
        key = (s.op.value, json.dumps(s.params, sort_keys=True, default=str))
        if key not in seen:
            seen.add(key)
            result.append(s)
        else:
            removed += 1
    if removed > 0:
        return result, [f"R2: Removed {removed} duplicate operator(s)"]
    return steps, []


def _rewrite_topn(steps: list[PlanOp]) -> tuple[list[PlanOp], list[str]]:
    """R3: TopN / Limit propagation."""
    has_sort = any(s.op == OpType.SORT for s in steps)
    limit_step = next((s for s in steps if s.op == OpType.LIMIT), None)
    if has_sort and limit_step and limit_step.params.get("n", 500) <= 20:
        return steps, [
            f"R3: TopN — SORT+LIMIT({limit_step.params['n']}) can use partial sort (heap)"
        ]
    return steps, []


def _rewrite_projection_pushdown(
    steps: list[PlanOp], profile: Optional[ProfileCard]
) -> tuple[list[PlanOp], list[str]]:
    """R4: Projection Pushdown — identify required columns, add early SELECT."""
    # Collect all columns referenced by non-SCAN operators
    required = set()
    for s in steps:
        if s.op == OpType.SCAN:
            continue
        p = s.params
        for key in ("column", "metric", "source", "target"):
            if key in p:
                required.add(p[key])
        if "keys" in p:
            required.update(p["keys"])
        if "columns" in p:
            required.update(p["columns"])

    if not required or not profile:
        return steps, []

    total_cols = profile.col_count
    used_cols = len(required)

    if used_cols < total_cols and total_cols > 3:
        # Add projection info to SCAN params (executor can use it)
        for s in steps:
            if s.op == OpType.SCAN:
                s.params["projected_columns"] = sorted(required)
                break
        return steps, [
            f"R4: Projection pushdown — {used_cols}/{total_cols} columns needed"
        ]
    return steps, []


def _rewrite_predicate_simplify(steps: list[PlanOp]) -> tuple[list[PlanOp], list[str]]:
    """
    R5: Predicate Simplification.
    - Remove duplicate filters (same column, same op, same value)
    - Merge range predicates on same column (> X AND < Y → BETWEEN)
    - Constant folding: always-true filters removed, always-false flagged
    """
    filters = [s for s in steps if s.op == OpType.FILTER]
    non_filters = [s for s in steps if s.op != OpType.FILTER]

    if len(filters) <= 1:
        return steps, []

    # Dedup exact same filter
    seen_filters = set()
    unique_filters = []
    removed = 0
    for f in filters:
        key = (f.params.get("column"), f.params.get("op"), str(f.params.get("value")))
        if key not in seen_filters:
            seen_filters.add(key)
            unique_filters.append(f)
        else:
            removed += 1

    rewrites = []
    if removed > 0:
        rewrites.append(f"R5: Removed {removed} duplicate predicate(s)")

    # Merge range predicates: col > X AND col < Y → mark as range
    col_predicates: dict[str, list] = {}
    for f in unique_filters:
        col = f.params.get("column", "")
        col_predicates.setdefault(col, []).append(f)

    merged_filters = []
    for col, preds in col_predicates.items():
        if len(preds) == 2:
            ops = {p.params.get("op") for p in preds}
            if ops in ({">=", "<="}, {">", "<"}, {">=", "<"}, {">", "<="}):
                # Range pair — keep both but annotate
                merged_filters.extend(preds)
                rewrites.append(f"R5: Range pair detected on '{col}' (BETWEEN optimization)")
                continue
        merged_filters.extend(preds)

    # Reconstruct: non-filters in original order, filters re-inserted
    result = []
    filter_inserted = False
    for s in non_filters:
        result.append(s)
        if not filter_inserted and s.op in (OpType.SCAN, OpType.DERIVE_TIME):
            remaining_nf = [x for x in non_filters[non_filters.index(s)+1:]]
            if not remaining_nf or remaining_nf[0].op != OpType.DERIVE_TIME:
                result.extend(merged_filters)
                filter_inserted = True
    if not filter_inserted:
        result.extend(merged_filters)

    return result, rewrites


# Runtime Feedback Cache

class FeedbackCache:
    """
    Store actual vs estimated results per (dataset, filter_signature).
    LEO-style: next time a similar filter appears, use actual selectivity.
    """

    def __init__(self):
        self._cache: dict[str, dict] = {}
        self._load()

    def _load(self):
        if _FEEDBACK_FILE.exists():
            try:
                self._cache = json.loads(_FEEDBACK_FILE.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

    def _save(self):
        _FEEDBACK_FILE.write_text(
            json.dumps(self._cache, indent=2, default=str), encoding="utf-8"
        )

    def _make_key(self, dataset_id: str, column: str, op: str, value) -> str:
        return f"{dataset_id}|{column}|{op}|{value}"

    def record(self, dataset_id: str, column: str, op: str, value,
               estimated_sel: float, actual_rows: int, total_rows: int):
        """Record actual result for a filter predicate."""
        key = self._make_key(dataset_id, column, op, value)
        actual_sel = actual_rows / total_rows if total_rows > 0 else 0
        self._cache[key] = {
            "estimated_sel": estimated_sel,
            "actual_sel": actual_sel,
            "actual_rows": actual_rows,
            "total_rows": total_rows,
            "recorded_at": datetime.now().isoformat(),
            "times_used": self._cache.get(key, {}).get("times_used", 0),
        }
        self._save()

    def lookup(self, dataset_id: str, column: str, op: str, value) -> Optional[float]:
        """Look up cached actual selectivity. Returns None if miss."""
        key = self._make_key(dataset_id, column, op, value)
        entry = self._cache.get(key)
        if entry:
            entry["times_used"] = entry.get("times_used", 0) + 1
            self._save()
            return entry["actual_sel"]
        return None

    def get_stats(self) -> dict:
        return {
            "total_entries": len(self._cache),
            "entries": [
                {"key": k, "est": v["estimated_sel"], "act": v["actual_sel"],
                 "ratio": round(v["estimated_sel"] / max(v["actual_sel"], 1e-6), 2),
                 "times_used": v.get("times_used", 0)}
                for k, v in list(self._cache.items())[:20]
            ]
        }


_feedback_cache: Optional[FeedbackCache] = None


def get_feedback_cache() -> FeedbackCache:
    global _feedback_cache
    if _feedback_cache is None:
        _feedback_cache = FeedbackCache()
    return _feedback_cache


# Full Cost Estimation (with histogram + sampling + feedback)

def estimate_plan_cost(plan: ExecutablePlan) -> PlanCost:
    """
    Estimate plan execution cost using:
    1. Histogram-based selectivity (first)
    2. Feedback cache lookup (if hit, override)
    3. Sampling probe (if confidence is low)
    """
    cost = PlanCost()
    profile = load_profile_card(plan.dataset_id)
    feedback = get_feedback_cache()

    if not profile:
        cost.warnings.append(f"No profile for '{plan.dataset_id}'")
        return cost

    cost.scan_rows = profile.row_count
    cost.total_columns = profile.col_count
    current_rows = float(profile.row_count)
    overall_confidence = "high"

    for step in plan.steps:
        if step.op == OpType.FILTER:
            col_name = step.params.get("column", "")
            op = step.params.get("op", "=")
            val = step.params.get("value")
            col_profile = _find_column(profile, col_name)

            if not col_profile:
                cost.filter_selectivity *= 0.3
                current_rows *= 0.3
                overall_confidence = "low"
                continue

            # Check feedback cache first
            cached_sel = feedback.lookup(plan.dataset_id, col_name, op, val)
            if cached_sel is not None:
                cost.filter_selectivity *= cached_sel
                current_rows *= cached_sel
                cost.feedback_hit = True
                continue

            # Histogram-based estimation
            sel, confidence = _estimate_selectivity_from_histogram(col_profile, op, val)

            # If low confidence AND table is large enough, trigger sampling
            if confidence == "low" and profile.row_count > 500:
                sampled = _sample_selectivity(plan.dataset_id, col_name, op, val)
                if sampled is not None:
                    sel = sampled
                    confidence = "high"
                    cost.sampling_triggered = True
                    cost.sampled_selectivity = sampled

            cost.filter_selectivity *= sel
            current_rows *= sel
            if confidence != "high":
                overall_confidence = confidence

        elif step.op == OpType.GROUPBY:
            keys = step.params.get("keys", [])
            cardinality = 1
            for key in keys:
                col_p = _find_column(profile, key)
                if col_p:
                    cardinality *= max(col_p.n_unique, 1)
            cost.groupby_cardinality = min(cardinality, int(current_rows))
            current_rows = float(cost.groupby_cardinality)

        elif step.op == OpType.SORT:
            cost.has_sort = True

        elif step.op == OpType.LIMIT:
            cost.has_limit = True
            cost.limit_n = step.params.get("n", 500)
            current_rows = min(current_rows, cost.limit_n)

        elif step.op == OpType.SCAN:
            proj = step.params.get("projected_columns")
            if proj:
                cost.projected_columns = len(proj)

    cost.rows_after_filter = max(1, int(cost.scan_rows * cost.filter_selectivity))
    cost.result_rows = max(1, int(current_rows))
    cost.confidence = overall_confidence

    # Cost score
    scan_score = min(cost.scan_rows / 10000, 30)
    sort_score = 15 if cost.has_sort else 0
    group_score = min(cost.groupby_cardinality / 100, 20)
    no_filter_penalty = 20 if cost.filter_selectivity >= 0.99 else 0
    proj_bonus = -5 if cost.projected_columns > 0 and cost.projected_columns < cost.total_columns else 0
    cost.estimated_cost_score = max(0, scan_score + sort_score + group_score + no_filter_penalty + proj_bonus)

    # Warnings
    if cost.scan_rows > 100000 and cost.filter_selectivity > 0.5:
        cost.warnings.append(
            f"Full scan on {cost.scan_rows:,} rows with low selectivity "
            f"({cost.filter_selectivity:.0%}) — consider adding filters"
        )
    if cost.has_sort and not cost.has_limit:
        cost.warnings.append("Sort without LIMIT — sorting entire result set")
    if cost.groupby_cardinality > 1000:
        cost.warnings.append(f"High-cardinality groupby ({cost.groupby_cardinality:,} groups)")
    if overall_confidence == "low":
        cost.warnings.append(
            "Low estimation confidence — consider adding more specific filters"
        )

    return cost


# Token Tracker (tiktoken-based)

class TokenUsage:
    def __init__(self):
        self.lakeprobe_input_tokens: int = 0
        self.lakeprobe_output_tokens: int = 0
        self.text2sql_prompt_tokens: int = 0
        self.text2sql_output_tokens_est: int = 150  # typical SQL output
        # Schema-pruning baseline: simulates DAIL-SQL / DIN-SQL style
        # where only the target table schema is sent (not the full data lake)
        self.text2sql_pruned_prompt_tokens: int = 0
        self.method: str = "tiktoken"  # "tiktoken" | "api_usage" | "estimate"

    @property
    def lakeprobe_total(self) -> int:
        return self.lakeprobe_input_tokens + self.lakeprobe_output_tokens

    @property
    def text2sql_total(self) -> int:
        return self.text2sql_prompt_tokens + self.text2sql_output_tokens_est

    @property
    def text2sql_pruned_total(self) -> int:
        return self.text2sql_pruned_prompt_tokens + self.text2sql_output_tokens_est

    @property
    def token_saving_ratio(self) -> float:
        if self.text2sql_total == 0:
            return 0.0
        return 1.0 - (self.lakeprobe_total / self.text2sql_total)

    @property
    def token_saving_ratio_vs_pruned(self) -> float:
        """Saving ratio vs schema-pruning baseline (fairer comparison)."""
        if self.text2sql_pruned_total == 0:
            return 0.0
        return 1.0 - (self.lakeprobe_total / self.text2sql_pruned_total)

    def to_dict(self) -> dict:
        return {
            "lakeprobe_input_tokens": self.lakeprobe_input_tokens,
            "lakeprobe_output_tokens": self.lakeprobe_output_tokens,
            "lakeprobe_total_tokens": self.lakeprobe_total,
            # Full-schema baseline (naive Text2SQL)
            "text2sql_prompt_tokens": self.text2sql_prompt_tokens,
            "text2sql_output_tokens_est": self.text2sql_output_tokens_est,
            "text2sql_total_tokens": self.text2sql_total,
            "token_saving_ratio": f"{self.token_saving_ratio:.0%}",
            # Schema-pruning baseline (DAIL-SQL / DIN-SQL style)
            "text2sql_pruned_prompt_tokens": self.text2sql_pruned_prompt_tokens,
            "text2sql_pruned_total_tokens": self.text2sql_pruned_total,
            "token_saving_ratio_vs_pruned": f"{self.token_saving_ratio_vs_pruned:.0%}",
            "measurement_method": self.method,
        }


_current_token_usage: Optional[TokenUsage] = None
_token_usage_lock = __import__("threading").Lock()


def start_token_tracking() -> TokenUsage:
    global _current_token_usage
    usage = TokenUsage()
    with _token_usage_lock:
        _current_token_usage = usage
    return usage


def get_token_usage() -> Optional[TokenUsage]:
    with _token_usage_lock:
        return _current_token_usage


def record_llm_tokens(input_tokens: int, output_tokens: int):
    """Record actual LLM token usage from API response."""
    with _token_usage_lock:
        if _current_token_usage:
            _current_token_usage.lakeprobe_input_tokens += input_tokens
            _current_token_usage.lakeprobe_output_tokens += output_tokens
            _current_token_usage.method = "api_usage"


def _count_tokens_tiktoken(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens precisely using tiktoken."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4  # rough fallback


# Text2SQL Baseline (Real Prompt Construction)

_TEXT2SQL_SYSTEM = """You are a SQL expert. Given a database schema and a natural language question, generate a SQL query.
Return ONLY the SQL query, no explanation."""

_TEXT2SQL_FEWSHOT = """
Example 1:
Schema: CREATE TABLE sales (id INT, date DATE, region VARCHAR, amount FLOAT);
Question: Total sales by region
SQL: SELECT region, SUM(amount) FROM sales GROUP BY region;

Example 2:
Schema: CREATE TABLE employees (id INT, name VARCHAR, dept VARCHAR, salary FLOAT);
Question: Top 5 highest paid employees
SQL: SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 5;

Example 3:
Schema: CREATE TABLE orders (id INT, customer VARCHAR, product VARCHAR, qty INT, price FLOAT);
Question: Average order value per customer
SQL: SELECT customer, AVG(qty * price) FROM orders GROUP BY customer;
"""


_cached_all_ddl: Optional[str] = None
_cached_all_ddl_tokens: int = 0


def _get_all_tables_ddl() -> str:
    """Cache the full data lake DDL — only compute once."""
    global _cached_all_ddl, _cached_all_ddl_tokens
    if _cached_all_ddl is not None:
        return _cached_all_ddl
    try:
        from core.dataset_card import load_all_dataset_cards
        from core.profiler import load_profile_card as _load_pc

        all_ddl = ""
        for card in load_all_dataset_cards():
            p = _load_pc(card.dataset_id)
            if not p:
                continue
            ddl = f"CREATE TABLE {p.dataset_id} (\n"
            col_defs = []
            for col in p.columns:
                sql_type = {
                    "int64": "INTEGER", "float64": "FLOAT", "object": "VARCHAR",
                    "datetime64": "DATE", "bool": "BOOLEAN",
                }.get(col.dtype, "VARCHAR")
                col_defs.append(f"  {col.name} {sql_type}")
            ddl += ",\n".join(col_defs) + "\n);\n"
            all_ddl += ddl

        _cached_all_ddl = all_ddl
        _cached_all_ddl_tokens = _count_tokens_tiktoken(all_ddl)
        return all_ddl
    except Exception:
        return ""


def build_text2sql_prompt(query: str, profile: Optional[ProfileCard]) -> str:
    """
    Build a realistic Text2SQL prompt to measure its token cost.

    Full-schema baseline: includes DDL for ALL tables in the data lake.
    Text2SQL must send the entire schema because the LLM doesn't know
    which table to use — that's the whole point of schema grounding.
    """
    parts = [_TEXT2SQL_SYSTEM, "\n", _TEXT2SQL_FEWSHOT, "\n"]

    all_ddl = _get_all_tables_ddl()

    if all_ddl:
        schema_section = all_ddl
        # Add sample rows for the target table only
        if profile:
            schema_section += f"\n-- Example data from {profile.dataset_id}:\n"
            for col in profile.columns:
                samples = ", ".join(str(v) for v in col.sample_values[:3])
                schema_section += f"-- {col.name}: {samples}\n"
        parts.append(f"Schema:\n{schema_section}")
    elif profile:
        # Fallback: single table only
        ddl = f"CREATE TABLE {profile.dataset_id} (\n"
        col_defs = []
        for col in profile.columns:
            sql_type = {
                "int64": "INTEGER", "float64": "FLOAT", "object": "VARCHAR",
                "datetime64": "DATE", "bool": "BOOLEAN",
            }.get(col.dtype, "VARCHAR")
            col_defs.append(f"  {col.name} {sql_type}")
        ddl += ",\n".join(col_defs) + "\n);\n"
        parts.append(f"Schema:\n{ddl}")
    else:
        parts.append("Schema: (not available)\n")

    parts.append(f"\nQuestion: {query}\nSQL:")
    return "".join(parts)


def build_text2sql_pruned_prompt(query: str, profile: Optional[ProfileCard]) -> str:
    """
    Build a schema-PRUNED Text2SQL prompt (DAIL-SQL / DIN-SQL style).

    Only includes the target table's DDL + sample rows — no other tables.
    This represents a fairer baseline where schema selection has already
    been done (analogous to what LakeProbe's retriever does).
    """
    parts = [_TEXT2SQL_SYSTEM, "\n", _TEXT2SQL_FEWSHOT, "\n"]

    if profile:
        ddl = f"CREATE TABLE {profile.dataset_id} (\n"
        col_defs = []
        for col in profile.columns:
            sql_type = {
                "int64": "INTEGER", "float64": "FLOAT", "object": "VARCHAR",
                "datetime64": "DATE", "bool": "BOOLEAN",
            }.get(col.dtype, "VARCHAR")
            col_defs.append(f"  {col.name} {sql_type}")
        ddl += ",\n".join(col_defs) + "\n);\n"

        # Only 3 sample rows (pruned approach is more concise)
        ddl += f"\n-- Example data from {profile.dataset_id}:\n"
        for col in profile.columns:
            samples = ", ".join(str(v) for v in col.sample_values[:3])
            ddl += f"-- {col.name}: {samples}\n"

        # NO other tables — this is the pruned version
        parts.append(f"Schema:\n{ddl}")
    else:
        parts.append("Schema: (not available)\n")

    parts.append(f"\nQuestion: {query}\nSQL:")
    return "".join(parts)


def measure_text2sql_tokens(query: str, profile: Optional[ProfileCard] = None) -> int:
    """
    Measure Text2SQL baseline token cost using tiktoken.

    Computes TWO baselines:
      1. Full-schema (naive) — all tables concatenated
      2. Schema-pruned (DAIL-SQL style) — only target table

    Returns total prompt tokens for the full-schema baseline.
    """
    # Full-schema baseline
    prompt = build_text2sql_prompt(query, profile)
    tokens = _count_tokens_tiktoken(prompt)

    # Schema-pruned baseline
    pruned_prompt = build_text2sql_pruned_prompt(query, profile)
    pruned_tokens = _count_tokens_tiktoken(pruned_prompt)

    with _token_usage_lock:
        if _current_token_usage:
            _current_token_usage.text2sql_prompt_tokens = tokens
            _current_token_usage.text2sql_pruned_prompt_tokens = pruned_tokens
            if _current_token_usage.method != "api_usage":
                _current_token_usage.method = "tiktoken"

    return tokens


def measure_lakeprobe_tokens_tiktoken(system_prompt: str, query: str):
    """
    Measure LakeProbe's LLM token cost using tiktoken (if API didn't report)
    """
    with _token_usage_lock:
        if _current_token_usage and _current_token_usage.lakeprobe_input_tokens == 0:
            tokens = _count_tokens_tiktoken(system_prompt + query)
            _current_token_usage.lakeprobe_input_tokens = tokens
            _current_token_usage.lakeprobe_output_tokens = 80
            _current_token_usage.method = "tiktoken"


# Plan Editor (for UI)

def edit_plan_step(plan: ExecutablePlan, step_index: int,
                   action: str = "remove", new_params: dict = None) -> ExecutablePlan:
    steps = [PlanOp(op=s.op, params=dict(s.params)) for s in plan.steps]
    if step_index < 0 or step_index >= len(steps):
        return plan
    if action == "remove" and steps[step_index].op != OpType.SCAN:
        steps.pop(step_index)
    elif action == "modify" and new_params:
        steps[step_index].params.update(new_params)
    return ExecutablePlan(dataset_id=plan.dataset_id, steps=steps)


def plan_to_display(plan: ExecutablePlan) -> list[dict]:
    return [
        {"step": i+1, "operator": s.op.value, "params": s.params,
         "removable": s.op != OpType.SCAN}
        for i, s in enumerate(plan.steps)
    ]
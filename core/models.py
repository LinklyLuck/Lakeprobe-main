"""
LakeProbe — System-wide Protocol Layer (Pydantic Models)
Function:
All data exchange between modules is handled entirely through the structs defined here.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# Enums
class IntentType(str, Enum):
    AGGREGATE = "aggregate"
    TREND = "trend"
    RANKING = "ranking"
    FILTER = "filter"
    COMPARISON = "comparison"
    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"
    LOOKUP = "lookup"
    UNKNOWN = "unknown"


class ColumnRole(str, Enum):
    MEASURE = "measure"
    DIMENSION = "dimension"
    TIME = "time"
    IDENTIFIER = "identifier"
    TEXT = "text"
    UNKNOWN = "unknown"


class OpType(str, Enum):
    SCAN = "scan"
    FILTER = "filter"
    DERIVE_TIME = "derive_time"
    GROUPBY = "groupby"
    AGGREGATE = "aggregate"
    SORT = "sort"
    LIMIT = "limit"
    JOIN = "join"
    SELECT = "select"


class AggFunc(str, Enum):
    SUM = "sum"
    AVG = "avg"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    COUNT_DISTINCT = "count_distinct"


# PartA — Query Intent
class FilterHint(BaseModel):
    field_hint: str
    op: str = "="
    value: Any = None


class QueryIntent(BaseModel):
    #Schema-agnostic semantic intent that does not include any specific column names.
    intent_type: IntentType = IntentType.UNKNOWN
    metric_hints: list[str] = Field(default_factory=list)
    dimension_hints: list[str] = Field(default_factory=list)
    filter_hints: list[FilterHint] = Field(default_factory=list)
    time_hints: list[str] = Field(default_factory=list)
    sort_hint: Optional[str] = None          # "asc" | "desc" | None
    limit_hint: Optional[int] = None
    agg_func_hint: Optional[AggFunc] = None
    raw_query: str = ""
    ambiguities: list[str] = Field(default_factory=list)
    needs_clarification: bool = False


# PartB — Profile & Dataset Cards
class ColumnProfile(BaseModel):
    name: str
    dtype: str                  # int64, float64, object, datetime64, bool ...
    missing_rate: float = 0.0
    unique_rate: float = 0.0
    n_unique: int = 0
    min_val: Optional[Any] = None
    max_val: Optional[Any] = None
    mean_val: Optional[float] = None
    top_values: list[Any] = Field(default_factory=list)
    sample_values: list[Any] = Field(default_factory=list)
    inferred_role: ColumnRole = ColumnRole.UNKNOWN
    pii_risk: bool = False
    # Frequency histogram
    histogram: list[tuple] = Field(default_factory=list)
    # Frequency Table
    value_counts: dict[str, int] = Field(default_factory=dict)


class ProfileCard(BaseModel):
    dataset_id: str
    file_path: str
    row_count: int = 0
    col_count: int = 0
    columns: list[ColumnProfile] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class DatasetCard(BaseModel):
    dataset_id: str
    purpose: str = ""
    domain: str = ""
    entities: list[str] = Field(default_factory=list)
    measure_columns: list[str] = Field(default_factory=list)
    dimension_columns: list[str] = Field(default_factory=list)
    time_columns: list[str] = Field(default_factory=list)
    row_count: int = 0
    column_names: list[str] = Field(default_factory=list)
    summary: str = ""
    summary_embedding: list[float] = Field(default_factory=list)  # dataset-level embedding


# Column Index
class ColumnIndexEntry(BaseModel):
    dataset_id: str
    column_name: str
    lexical_key: str            # lower-cased, stripped
    aliases: list[str] = Field(default_factory=list)
    inferred_role: ColumnRole = ColumnRole.UNKNOWN
    dtype: str = ""
    stats_fingerprint: dict = Field(default_factory=dict)
    embedding: Optional[list[float]] = None   # optional vector


# Retrieval Candidates
class ColumnCandidate(BaseModel):
    dataset_id: str
    column_name: str
    role: ColumnRole = ColumnRole.UNKNOWN
    score: float = 0.0
    evidence: list[str] = Field(default_factory=list)
    dtype: str = ""


class RetrievalResult(BaseModel):
    dataset_candidates: list[str] = Field(default_factory=list)
    metric_candidates: list[ColumnCandidate] = Field(default_factory=list)
    dimension_candidates: list[ColumnCandidate] = Field(default_factory=list)
    time_candidates: list[ColumnCandidate] = Field(default_factory=list)
    filter_candidates: list[ColumnCandidate] = Field(default_factory=list)


# Fusion — Binding & Plan
class BindingEntry(BaseModel):
    hint: str
    column: str
    dataset_id: str = ""
    score: float = 0.0
    zone: str = "accept"           # "accept" | "uncertain" | "reject"
    evidence: list[str] = Field(default_factory=list)


class BindingResult(BaseModel):
    dataset_id: str
    metric_bindings: list[BindingEntry] = Field(default_factory=list)
    dimension_bindings: list[BindingEntry] = Field(default_factory=list)
    time_bindings: list[BindingEntry] = Field(default_factory=list)
    filter_bindings: list[BindingEntry] = Field(default_factory=list)
    blocked_candidates: list[dict] = Field(default_factory=list)


class PlanOp(BaseModel):
    op: OpType
    params: dict = Field(default_factory=dict)


class ExecutablePlan(BaseModel):
    dataset_id: str
    steps: list[PlanOp] = Field(default_factory=list)


# Audit Trail
class AuditRecord(BaseModel):
    query_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    raw_query: str = ""
    query_intent: dict = Field(default_factory=dict)
    candidates: dict = Field(default_factory=dict)
    blocked_candidates: list[dict] = Field(default_factory=list)
    final_binding: dict = Field(default_factory=dict)
    executable_plan: dict = Field(default_factory=dict)
    execution_summary: dict = Field(default_factory=dict)
    user_override: Optional[dict] = None
    clarification_history: list[str] = Field(default_factory=list)


# Discovery — Desired Schema + Column Match
class DesiredColumn(BaseModel):
    """LLM 生成的理想列描述。"""
    name: str                        # e.g., "alcohol_content"
    description: str = ""            # e.g., "percentage of alcohol by volume"
    expected_dtype: str = ""         # "numeric" | "categorical" | "datetime" | "text"
    role: str = ""                   # "feature" | "target" | "identifier" | "time"
    importance: str = "required"     # "required" | "optional" | "nice_to_have"


class DesiredSchema(BaseModel):
    """An ideal column description generated by an LLM"""
    task_type: str = ""              # "classification" | "regression" | "clustering" | "analysis" | "exploration"
    domain: str = ""                 # "wine" | "housing" | "finance" ...
    target_description: str = ""     # "predict wine quality score"
    desired_columns: list[DesiredColumn] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)  # "at least 1000 rows", "must have time column"


class ColumnMatch(BaseModel):
    """The results of matching an actual column with an ideal column."""
    desired: str                     # List of Ideal Candidates
    actual_column: str               # Column names actually matched
    actual_dataset: str
    match_score: float = 0.0
    evidence: list[str] = Field(default_factory=list)


class DatasetDiscoveryResult(BaseModel):
    """The discovery score results for a dataset."""
    dataset_id: str
    overall_score: float = 0.0
    matched_columns: list[ColumnMatch] = Field(default_factory=list)
    coverage: float = 0.0            # How many of the desired columns are covered
    row_count: int = 0
    domain: str = ""
    summary: str = ""


# Join Discovery
class JoinCandidate(BaseModel):
    """两个列之间的 joinability 候选。"""
    left_dataset: str
    left_column: str
    right_dataset: str
    right_column: str
    overlap_ratio: float = 0.0       # |intersection| / min(|left|, |right|)
    left_distinct: int = 0
    right_distinct: int = 0
    intersection_size: int = 0
    sample_overlapping_values: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)

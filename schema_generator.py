"""
LakeProbe — Schema Generator + Column IR Matching

User Query → LLM generates "Desired Schema" → IR matches against ProfileCard/DatasetCard
                                                → Ranked datasets + column matches

Example:
  Query: "I want a red wine dataset to predict wine quality"
  → DesiredSchema:
      task_type: "classification"
      domain: "wine"
      target: "wine quality score"
      columns: [
        {name: "quality", role: "target", dtype: "numeric"},
        {name: "alcohol", role: "feature", dtype: "numeric"},
        {name: "acidity", role: "feature", dtype: "numeric"},
        {name: "sugar", role: "feature", dtype: "numeric"},
        ...
      ]
  → IR matches "quality" against all indexed columns → finds wine.csv:quality (score=0.95)
"""

from __future__ import annotations

import json
import logging
import os
import re
from difflib import SequenceMatcher
from typing import Optional

import numpy as np

from core.models import (
    DesiredColumn, DesiredSchema, ColumnMatch,
    DatasetDiscoveryResult, ProfileCard, ColumnRole,
)

logger = logging.getLogger(__name__)

# 1. LLM Schema Generation

_SCHEMA_SYSTEM_PROMPT = """\
You are a data schema designer. Given a user's natural language request about what data they need,
generate a STRUCTURED description of the ideal dataset schema.

Return STRICTLY valid JSON (no markdown fences):
{
  "task_type": "classification|regression|clustering|analysis|exploration",
  "domain": "<domain keyword, e.g. wine, housing, finance>",
  "target_description": "<what the user wants to predict/analyze>",
  "desired_columns": [
    {
      "name": "<column name hint>",
      "description": "<what this column represents>",
      "expected_dtype": "numeric|categorical|datetime|text",
      "role": "feature|target|identifier|time",
      "importance": "required|optional|nice_to_have"
    }
  ],
  "constraints": ["<e.g., at least 1000 rows>", "<must have numeric target>"]
}

Generate 5-15 columns that would be useful for the described task.
Be specific to the domain. For example, for wine quality prediction,
include columns like alcohol, acidity, sulfur_dioxide, etc.

Examples:

Q: "I want a dataset to predict house prices"
A: {"task_type":"regression","domain":"housing","target_description":"predict house sale price","desired_columns":[{"name":"price","description":"sale price of the house","expected_dtype":"numeric","role":"target","importance":"required"},{"name":"area","description":"living area in sq ft","expected_dtype":"numeric","role":"feature","importance":"required"},{"name":"bedrooms","description":"number of bedrooms","expected_dtype":"numeric","role":"feature","importance":"required"},{"name":"location","description":"neighborhood or zip code","expected_dtype":"categorical","role":"feature","importance":"required"},{"name":"year_built","description":"year the house was built","expected_dtype":"numeric","role":"feature","importance":"optional"}],"constraints":["at least 500 rows","must have numeric price column"]}

Q: "Find me customer churn data for telecom"
A: {"task_type":"classification","domain":"telecom","target_description":"predict whether customer will churn","desired_columns":[{"name":"churn","description":"whether customer churned (yes/no)","expected_dtype":"categorical","role":"target","importance":"required"},{"name":"tenure","description":"months as customer","expected_dtype":"numeric","role":"feature","importance":"required"},{"name":"monthly_charges","description":"monthly bill amount","expected_dtype":"numeric","role":"feature","importance":"required"},{"name":"contract","description":"contract type (month-to-month, yearly)","expected_dtype":"categorical","role":"feature","importance":"required"},{"name":"total_charges","description":"total amount billed","expected_dtype":"numeric","role":"feature","importance":"optional"}],"constraints":["must have binary churn label"]}
"""


def generate_desired_schema(query: str) -> DesiredSchema:
    """
    Use LLM to generate a desired schema from user query.
    Falls back to keyword extraction if LLM unavailable.
    """
    # Try LLM
    schema = _call_llm_schema(query)
    if schema:
        return schema

    # Fallback: keyword extraction
    return _fallback_schema(query)


def _call_llm_schema(query: str) -> Optional[DesiredSchema]:
    """Call LLM to generate desired schema."""
    try:
        import openai
        from config import LLM_MODEL, LLM_TEMPERATURE, LLM_API_BASE, LLM_API_KEY

        api_key = LLM_API_KEY or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        api_base = LLM_API_BASE or os.getenv("LLM_API_BASE")
        if not api_key:
            return None

        client_kwargs = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base

        client = openai.OpenAI(**client_kwargs, timeout=30)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": _SCHEMA_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )

        # Token tracking
        try:
            from core.plan_optimizer import record_llm_tokens
            if resp.usage:
                record_llm_tokens(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        except Exception:
            pass

        text = resp.choices[0].message.content.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        raw = json.loads(text)

        cols = [DesiredColumn(**c) for c in raw.get("desired_columns", [])]
        return DesiredSchema(
            task_type=raw.get("task_type", ""),
            domain=raw.get("domain", ""),
            target_description=raw.get("target_description", ""),
            desired_columns=cols,
            constraints=raw.get("constraints", []),
        )
    except Exception as e:
        logger.warning(f"[SchemaGen] LLM failed: {e}")
        return None


def _fallback_schema(query: str) -> DesiredSchema:
    """Keyword-based fallback schema generation."""
    q = query.lower()
    cols = []
    domain = ""

    # Extract domain keywords
    stop = {"i", "want", "a", "the", "to", "for", "of", "in", "and", "or", "me",
            "find", "show", "get", "give", "need", "looking", "dataset", "data",
            "predict", "classify", "cluster", "about", "with", "that", "can"}
    tokens = [t for t in q.replace(",", " ").replace(".", " ").split()
              if len(t) > 2 and t not in stop]
    if tokens:
        domain = tokens[0]

    # Detect task type
    task = "exploration"
    if any(w in q for w in ["predict", "regression", "forecast"]):
        task = "regression"
    elif any(w in q for w in ["classify", "classification", "detect", "churn"]):
        task = "classification"
    elif any(w in q for w in ["cluster", "segment", "group"]):
        task = "clustering"

    # Generate generic columns from keywords
    for t in tokens[:8]:
        cols.append(DesiredColumn(
            name=t,
            description=f"related to {t}",
            expected_dtype="numeric",
            role="feature",
            importance="optional",
        ))

    return DesiredSchema(
        task_type=task, domain=domain,
        target_description=query,
        desired_columns=cols,
    )


# IR Column Matching
def match_schema_to_datasets(
        desired: DesiredSchema,
    raw_query: str = "",
    pre_filter_top_k: int = 10,
) -> list[DatasetDiscoveryResult]:
    """
    Match desired schema against indexed datasets.

    Two-stage approach:
      Stage 1: Embedding pre-filter — use query embedding vs dataset summary
               embeddings to narrow down candidates (handles dirty table names)
      Stage 2: Fine-grained column matching — only on pre-filtered datasets

    This is critical for dirty data: table name might be "tbl_0xf3a2",
    but the summary embedding captures column content + sample values.
    """
    from core.dataset_card import load_all_dataset_cards, load_column_index
    from core.profiler import load_profile_card
    from core.embedding_engine import get_encoder

    all_cards = load_all_dataset_cards()
    if not all_cards:
        return []

    # Stage 1: Embedding Pre-Filter
    encoder = get_encoder()

    # Build query embedding text from desired schema
    query_parts = [raw_query] if raw_query else []
    if desired.domain:
        query_parts.append(desired.domain)
    if desired.target_description:
        query_parts.append(desired.target_description)
    for dc in desired.desired_columns:
        query_parts.append(f"{dc.name} {dc.description}")
    query_text = " | ".join(query_parts)
    query_vec = encoder.encode([query_text])[0]

    # Score all datasets by embedding similarity
    scored_cards = []
    for card in all_cards:
        sim = 0.0
        if card.summary_embedding:
            import numpy as np
            card_vec = np.array(card.summary_embedding, dtype=np.float32)
            sim = float(encoder.similarity(query_vec, card_vec.reshape(1, -1))[0])

        # Also add keyword bonus (fallback for cards without embedding)
        keyword_bonus = 0.0
        query_lower = raw_query.lower() if raw_query else ""
        ds_text = f"{card.dataset_id} {card.domain} {card.summary}".lower()
        ds_keywords = [t for t in card.dataset_id.lower().replace("_", " ").replace("-", " ").split()
                       if len(t) > 2 and not t.isdigit()]
        for kw in ds_keywords:
            if kw in query_lower:
                keyword_bonus += 0.3
                break
        if desired.domain and desired.domain.lower() in ds_text:
            keyword_bonus += 0.2

        scored_cards.append((card, sim + keyword_bonus))

    scored_cards.sort(key=lambda x: x[1], reverse=True)

    # Take top-k for fine-grained matching
    top_cards = [card for card, _ in scored_cards[:pre_filter_top_k]]
    logger.info(f"[Discovery] Pre-filter: {len(all_cards)} datasets → {len(top_cards)} candidates "
                f"(top: {[c.dataset_id for c in top_cards[:3]]})")

    # Stage 2: Fine-Grained Column Matching
    # Pre-encode all desired column query vectors (ONCE, not per dataset)
    desired_query_vecs = {}
    try:
        from core.embedding_engine import build_hint_text
        hint_texts = []
        hint_keys = []
        for dc in desired.desired_columns:
            dn = dc.name.lower().strip()
            hint_texts.append(build_hint_text(dn, hint_type=dc.role))
            hint_keys.append(dn)
        if hint_texts:
            all_vecs = encoder.encode(hint_texts)
            for i, key in enumerate(hint_keys):
                desired_query_vecs[key] = all_vecs[i]
    except Exception:
        pass

    results: list[DatasetDiscoveryResult] = []

    for card in top_cards:
        profile = load_profile_card(card.dataset_id)
        col_index = load_column_index(card.dataset_id)
        if not profile or not col_index:
            continue

        # Pre-load vectors for this dataset ONCE
        try:
            from core.embedding_engine import load_vectors
            vec_data = load_vectors(card.dataset_id)
        except Exception:
            vec_data = None

        matches: list[ColumnMatch] = []
        total_score = 0.0

        # Domain bonus
        domain_bonus = 0.0
        ds_text = f"{card.dataset_id} {card.domain} {card.summary}".lower()
        if desired.domain and desired.domain.lower() in ds_text:
            domain_bonus = 2.0
        query_lower = raw_query.lower() if raw_query else ""
        ds_keywords = [t for t in card.dataset_id.lower().replace("_", " ").replace("-", " ").split()
                       if len(t) > 2 and not t.isdigit()]
        for kw in ds_keywords:
            if kw in query_lower:
                domain_bonus += 1.5
                break

        for desired_col in desired.desired_columns:
            # Get pre-encoded query vector for this desired column
            q_vec = desired_query_vecs.get(desired_col.name.lower().strip())
            best_match = _find_best_column_match(
                desired_col, col_index, profile, card.dataset_id,
                vec_data=vec_data, query_vec=q_vec,
            )
            if best_match:
                matches.append(best_match)
                total_score += best_match.match_score

        n_desired = len(desired.desired_columns) or 1
        coverage = len(matches) / n_desired

        # Constraint checking
        constraint_penalty = 0.0
        for constraint in desired.constraints:
            cl = constraint.lower()
            if "at least" in cl and "row" in cl:
                try:
                    num = int(re.search(r"\d+", cl).group())
                    if card.row_count < num:
                        constraint_penalty += 0.5
                except Exception:
                    pass

        overall = (total_score / n_desired + domain_bonus + coverage) - constraint_penalty

        results.append(DatasetDiscoveryResult(
            dataset_id=card.dataset_id,
            overall_score=round(overall, 3),
            matched_columns=matches,
            coverage=round(coverage, 3),
            row_count=card.row_count,
            domain=card.domain,
            summary=card.summary,
        ))

    results.sort(key=lambda r: r.overall_score, reverse=True)
    return results[:10]


def _find_best_column_match(
    desired: DesiredColumn,
    col_index: list,
    profile: ProfileCard,
    dataset_id: str,
    vec_data: dict = None,
    query_vec=None,
) -> Optional[ColumnMatch]:
    """
    Find the best actual column matching a desired column.

    vec_data and query_vec are pre-loaded/pre-encoded outside the loop
    to avoid repeated disk I/O and embedding API calls.
    """
    best_score = 0.0
    best_match = None

    desired_name = desired.name.lower().strip()
    desired_desc = desired.description.lower()

    for entry in col_index:
        score = 0.0
        evidence = []

        col_name_lower = entry.column_name.lower()
        lexical_key = entry.lexical_key

        # 1. Lexical name match
        if desired_name == lexical_key or desired_name == col_name_lower:
            score += 1.0
            evidence.append("name exact match")
        elif desired_name in lexical_key or lexical_key in desired_name:
            score += 0.6
            evidence.append("name partial match")
        else:
            ratio = SequenceMatcher(None, desired_name, lexical_key).ratio()
            if ratio > 0.6:
                score += ratio * 0.5
                evidence.append(f"name fuzzy={ratio:.2f}")

        # 2. Alias match
        for alias in entry.aliases:
            alias_lower = alias.lower()
            if desired_name == alias_lower:
                score += 0.8
                evidence.append(f"alias exact={alias}")
                break
            elif desired_name in alias_lower or alias_lower in desired_name:
                score += 0.4
                evidence.append(f"alias partial={alias}")
                break
            if alias_lower in desired_desc:
                score += 0.3
                evidence.append(f"desc↔alias={alias}")
                break

        # 3. Dtype compatibility
        dtype_map = {
            "numeric": {"int64", "float64"},
            "categorical": {"object"},
            "datetime": {"datetime64"},
            "text": {"object"},
        }
        expected_dtypes = dtype_map.get(desired.expected_dtype, set())
        if expected_dtypes and entry.dtype in expected_dtypes:
            score += 0.3
            evidence.append(f"dtype compatible ({entry.dtype})")
        elif expected_dtypes and entry.dtype not in expected_dtypes:
            score -= 0.2
            evidence.append(f"dtype mismatch (want {desired.expected_dtype}, got {entry.dtype})")

        # 4. Role compatibility
        role_map = {
            "feature": {ColumnRole.MEASURE, ColumnRole.DIMENSION},
            "target": {ColumnRole.MEASURE, ColumnRole.DIMENSION},
            "identifier": {ColumnRole.IDENTIFIER},
            "time": {ColumnRole.TIME},
        }
        expected_roles = role_map.get(desired.role, set())
        if expected_roles and entry.inferred_role in expected_roles:
            score += 0.2
            evidence.append(f"role compatible ({entry.inferred_role.value})")

        # 5. Embedding similarity (pre-loaded, no disk I/O here)
        if vec_data is not None and query_vec is not None:
            try:
                col_names = list(vec_data["column_names"])
                if entry.column_name in col_names:
                    from core.embedding_engine import get_encoder
                    encoder = get_encoder()
                    col_idx = col_names.index(entry.column_name)
                    col_vec = vec_data["vectors"][col_idx]
                    sim = float(encoder.similarity(query_vec, col_vec.reshape(1, -1))[0])
                    if sim > 0.3:
                        score += sim * 0.4
                        evidence.append(f"embedding_sim={sim:.3f}")
            except Exception:
                pass

        if score > best_score:
            best_score = score
            best_match = ColumnMatch(
                desired=desired.name,
                actual_column=entry.column_name,
                actual_dataset=dataset_id,
                match_score=round(score, 4),
                evidence=evidence,
            )

    if best_match and best_match.match_score > 0.2:
        return best_match
    return None

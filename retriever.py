"""
LakeProbe — PartB: Hybrid Sparse-Dense Two-Stage Retriever

Stage 1: Dataset Retrieval — hybrid scoring selects Top-K datasets
Stage 2: Column Retrieval  — sparse (lexical + alias + role + dtype + stats)
                             + dense (embedding cosine similarity)
                             → Weighted Fusion / RRF → final ranking

Sparse signals: lexical match, alias match, role match, dtype compatibility, stats evidence
Dense signals: TF-IDF char n-gram SVD (lightweight) or sentence-transformers (production-grade)
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Optional

import numpy as np

from core.models import (
    ColumnCandidate,
    ColumnRole,
    QueryIntent,
    RetrievalResult,
)
from core.dataset_card import (
    load_all_dataset_cards,
    load_column_index,
    DatasetCard,
    ColumnIndexEntry,
)
from core.embedding_engine import (
    get_encoder,
    build_hint_text,
    load_vectors,
)
from config import (
    ALIAS_LEXICON,
    ALIAS_REVERSE,
    COLUMN_TOP_K,
    DATASET_TOP_K,
    MIN_CANDIDATE_SCORE,
    RETRIEVAL_WEIGHTS,
)

logger = logging.getLogger(__name__)


# Stage 1: Dataset Retrieval (sparse)
def _score_dataset(card: DatasetCard, intent: QueryIntent) -> float:
    """Score a single DatasetCard to measure its relevance to the intent."""
    score = 0.0
    all_hints = (intent.metric_hints + intent.dimension_hints +
                 [fh.field_hint for fh in intent.filter_hints] + intent.time_hints)

    # Direct match: raw query vs dataset_id
    # "I want wine dataset" → dataset_id "wine" → strong hit
    query_lower = intent.raw_query.lower()
    ds_id_lower = card.dataset_id.lower()
    # Keywords in dataset_id (remove numeric suffixes)
    ds_keywords = [t for t in ds_id_lower.replace("_", " ").replace("-", " ").split()
                   if len(t) > 2 and not t.isdigit()]
    for kw in ds_keywords:
        if kw in query_lower:
            score += 2.0
            break

    # Match nouns in the query against dataset summary/domain
    summary_lower = card.summary.lower()
    domain_lower = card.domain.lower()
    # Extract non-stopword tokens from the query
    stop_words = {"i", "want", "a", "the", "to", "of", "in", "for", "and", "or",
                  "show", "me", "find", "get", "give", "dataset", "data", "predict",
                  "is", "are", "was", "were", "be", "have", "has", "do", "does",
                  "what", "which", "how", "my", "this", "that", "with", "from", "by"}
    query_tokens = [t for t in query_lower.replace(",", " ").replace(".", " ").split()
                    if len(t) > 2 and t not in stop_words]
    for qt in query_tokens:
        if qt in ds_id_lower or qt in summary_lower or qt in domain_lower:
            score += 1.0

    if not all_hints and score > 0:
        return min(score / max(len(query_tokens), 1), 1.0)

    if not all_hints:
        return 0.1

    col_names_lower = [c.lower() for c in card.column_names]

    for hint in all_hints:
        hint_lower = hint.lower()

        # Exact column name match
        if hint_lower in col_names_lower:
            score += 1.0
            continue

        # Fuzzy column name match
        best_ratio = max(
            (SequenceMatcher(None, hint_lower, cn).ratio() for cn in col_names_lower),
            default=0,
        )
        if best_ratio > 0.7:
            score += best_ratio * 0.8

        # Summary text containment
        if hint_lower in summary_lower:
            score += 0.5

        # Alias check
        canonical = ALIAS_REVERSE.get(hint_lower, hint_lower)
        aliases = ALIAS_LEXICON.get(canonical, [])
        for alias in [canonical] + aliases:
            if alias.lower() in summary_lower or alias.lower() in col_names_lower:
                score += 0.6
                break

    max_possible = len(all_hints) * 1.0 + len(query_tokens) * 1.0
    return min(score / max(max_possible, 1), 1.0)


def _semantic_dataset_score(card: DatasetCard, intent: QueryIntent) -> float:
    """
    Compute semantic similarity between query and dataset summary embedding.
    Uses the pre-computed summary_embedding in DatasetCard.
    """
    if not card.summary_embedding:
        return 0.0

    try:
        encoder = get_encoder()
        # Build query text from intent hints
        query_text = build_hint_text(intent)
        if not query_text.strip():
            query_text = intent.raw_query

        query_vec = encoder.encode([query_text])[0]
        summary_vec = np.array(card.summary_embedding, dtype=np.float32)

        # Cosine similarity
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        s_norm = summary_vec / (np.linalg.norm(summary_vec) + 1e-10)
        sim = float(np.dot(q_norm, s_norm))
        return max(0.0, sim)
    except Exception:
        return 0.0


def rerank_datasets(dataset_ids: list[str], intent: QueryIntent) -> list[str]:
    """
    Table Rerank: re-score Top-K candidates using semantic similarity
    between query and table summary embeddings.

    This addresses the critical gap between Top-3 recall (95.5%) and
    Strict accuracy (54%) observed in BIRD benchmarks. The reranker
    uses the already-computed summary_embedding in DatasetCard for
    zero additional LLM cost.
    """
    if len(dataset_ids) <= 1:
        return dataset_ids

    cards = load_all_dataset_cards()
    card_map = {c.dataset_id: c for c in cards}

    reranked = []
    for ds_id in dataset_ids:
        card = card_map.get(ds_id)
        if not card:
            reranked.append((ds_id, 0.0))
            continue

        # Combine lexical score with semantic score
        lexical = _score_dataset(card, intent)
        semantic = _semantic_dataset_score(card, intent)

        # Weighted combination: semantic gets higher weight for disambiguation
        combined = 0.4 * lexical + 0.6 * semantic
        reranked.append((ds_id, combined))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return [ds_id for ds_id, _ in reranked]


def retrieve_datasets(intent: QueryIntent, domain_prior: str = None) -> list[str]:
    """
    Stage 1: Return Top-K relevant dataset IDs.

    Pipeline:
      Step 1 — Sparse-only scoring + domain boost → broad candidate pool
      Step 2 — Semantic rerank on Top-K → final ordering

    Sparse and semantic play different roles:
      - Step 1 uses only lexical/alias/domain signals for coarse filtering
        (fast, broad coverage)
      - Step 2 performs semantic reranking on shortlisted candidates
        (more precise, better discrimination)

    This avoids the previous problem where semantic signals were mixed
    into both stages and caused conflicting weights.
    """
    from config import DOMAIN_BOOST_WEIGHT

    cards = load_all_dataset_cards()
    if not cards:
        return []

    scored = []
    for card in cards:
        # Step 1: Sparse-only scoring (NO semantic here — leave it to rerank)
        base_score = _score_dataset(card, intent)

        # Domain routing soft boost (NOT a filter)
        if domain_prior and domain_prior not in ("general", "unknown", "off"):
            if card.domain.lower() == domain_prior.lower():
                base_score += DOMAIN_BOOST_WEIGHT
            elif domain_prior.lower() in card.dataset_id.lower():
                base_score += DOMAIN_BOOST_WEIGHT * 0.5
        scored.append((card.dataset_id, base_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    # Broad candidate pool for reranking
    candidates = [sid for sid, sc in scored if sc >= MIN_CANDIDATE_SCORE]
    if not candidates and scored:
        candidates = [scored[0][0]]
    top_k = candidates[:max(DATASET_TOP_K, 5)]

    # Step 2: Semantic rerank on top candidates
    reranked = rerank_datasets(top_k, intent)
    return reranked[:DATASET_TOP_K]


# Stage 2: Hybrid Column Retrieval

# 2A. Sparse Scorer (lexical + structural)

def _sparse_score(entry: ColumnIndexEntry, hint: str,
                   expected_role: ColumnRole | None = None) -> tuple[float, list[str]]:
    """
      Sparse signal scoring, including 5 channels:
      1. Lexical match     — exact / partial
      2. Alias match       — exact / fuzzy
      3. Role match        — role consistency
      4. Dtype compatibility
      5. Stats evidence    — statistical characteristics
    """
    hint_lower = hint.lower().strip()
    score = 0.0
    evidence: list[str] = []

    # Lexical match
    if hint_lower == entry.lexical_key:
        score += 1.0
        evidence.append("lexical exact match")
    elif hint_lower in entry.lexical_key or entry.lexical_key in hint_lower:
        score += 0.6
        evidence.append("lexical partial match")

    # Alias match
    for alias in entry.aliases:
        if alias.lower() == hint_lower:
            score += 0.8
            evidence.append(f"alias exact={alias}")
            break
        ratio = SequenceMatcher(None, alias.lower(), hint_lower).ratio()
        if ratio > 0.8:
            score += 0.5
            evidence.append(f"alias fuzzy={alias} ({ratio:.2f})")
            break

    # Role match
    if expected_role and entry.inferred_role == expected_role:
        score += 0.4
        evidence.append(f"role={expected_role.value}")
    elif expected_role and entry.inferred_role != expected_role:
        score -= 0.2
        evidence.append(f"role mismatch: want {expected_role.value}, got {entry.inferred_role.value}")

    # Dtype compatibility
    if expected_role == ColumnRole.MEASURE and entry.dtype in ("int64", "float64"):
        score += 0.3
        evidence.append(f"dtype={entry.dtype}")
    elif expected_role == ColumnRole.DIMENSION and entry.dtype == "object":
        score += 0.2
        evidence.append("dtype=categorical")
    elif expected_role == ColumnRole.TIME and entry.dtype == "datetime64":
        score += 0.3
        evidence.append("dtype=datetime")

    # Statistics evidence
    stats = entry.stats_fingerprint
    if expected_role == ColumnRole.DIMENSION:
        ur = stats.get("unique_rate", 1)
        if ur < 0.3:
            score += 0.2
            evidence.append(f"unique_rate={ur}")
    elif expected_role == ColumnRole.MEASURE:
        ur = stats.get("unique_rate", 0)
        if ur > 0.3:
            score += 0.1
            evidence.append(f"unique_rate={ur}")

    return score, evidence


# 2B. Dense Scorer (embedding similarity)

def _dense_scores_for_hint(
    hint: str,
    hint_type: str,
    dataset_id: str,
    agg_func: str = "",
) -> dict[str, float]:
    """
    Compute dense similarity scores for all columns for a given hint.

    Uses ANN index when available (for O(sqrt(N)) search across all datasets),
    and falls back to per-dataset brute-force search.

    Returns {column_name: cosine_similarity}
    """
    encoder = get_encoder()
    hint_text = build_hint_text(hint, hint_type=hint_type, agg_func=agg_func)
    query_vec = encoder.encode([hint_text])[0]  # [D]

    # Try ANN index first (fast, cross-dataset)
    try:
        from core.embedding_engine import get_ann_index
        ann = get_ann_index()
        if ann.size > 0:
            results = ann.search(query_vec, top_k=COLUMN_TOP_K * 2,
                                 dataset_filter=[dataset_id])
            if results:
                return {col_name: score for _, col_name, score in results}
    except Exception:
        pass

    # Fallback: per-dataset brute-force
    vec_data = load_vectors(dataset_id)
    if vec_data is None:
        return {}

    corpus_mat = vec_data["vectors"]  # [N, D]
    col_names = vec_data["column_names"]

    sims = encoder.similarity(query_vec, corpus_mat)  # [N]

    return {col_names[i]: float(sims[i]) for i in range(len(col_names))}


# 2C. Hybrid Fusion: Weighted Linear
def _hybrid_rank_columns(
    entries: list[ColumnIndexEntry],
    hint: str,
    expected_role: ColumnRole | None,
    dataset_id: str,
    agg_func: str = "",
) -> list[ColumnCandidate]:
    """
    Hybrid sparse + dense column-level ranking.

    Fusion strategy: weighted linear combination
      final_score = w_sparse * norm(sparse_score) + w_dense * norm(dense_score)

    sparse_score is min-max normalized to [0, 1].
    dense_score (cosine) in [-1, 1] is mapped to [0, 1].
    """
    w_sparse = RETRIEVAL_WEIGHTS.get("sparse", 0.5)
    w_dense = RETRIEVAL_WEIGHTS.get("dense", 0.5)

    # Sparse scores
    sparse_results: dict[str, tuple[float, list[str]]] = {}
    for entry in entries:
        sc, ev = _sparse_score(entry, hint, expected_role)
        sparse_results[entry.column_name] = (sc, ev)

    # Dense scores
    hint_type = ""
    if expected_role == ColumnRole.MEASURE:
        hint_type = "measure"
    elif expected_role == ColumnRole.DIMENSION:
        hint_type = "dimension"
    elif expected_role == ColumnRole.TIME:
        hint_type = "time"

    dense_map = _dense_scores_for_hint(hint, hint_type, dataset_id, agg_func)

    # Min-max normalize sparse
    sp_values = [v[0] for v in sparse_results.values()]
    s_min = min(sp_values) if sp_values else 0
    s_max = max(sp_values) if sp_values else 1
    s_range = s_max - s_min if s_max > s_min else 1.0

    # Fuse
    candidates: list[ColumnCandidate] = []
    for entry in entries:
        cn = entry.column_name

        sp_score, sp_evidence = sparse_results.get(cn, (0.0, []))
        sp_norm = (sp_score - s_min) / s_range

        # Dense: cosine ∈ [-1, 1] → [0, 1]
        dn_raw = dense_map.get(cn, 0.0)
        dn_norm = (dn_raw + 1.0) / 2.0

        final_score = w_sparse * sp_norm + w_dense * dn_norm

        # Build evidence
        evidence = sp_evidence.copy()
        if dn_raw > 0.3:
            evidence.append(f"dense_sim={dn_raw:.3f}")
        elif dn_raw > 0.0:
            evidence.append(f"dense_sim={dn_raw:.3f} (weak)")

        if final_score > 0:
            candidates.append(ColumnCandidate(
                dataset_id=dataset_id,
                column_name=cn,
                role=entry.inferred_role,
                score=round(final_score, 4),
                evidence=evidence,
                dtype=entry.dtype,
            ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:COLUMN_TOP_K]


# 2D. Reciprocal Rank Fusion (alternative)
def _rrf_rank_columns(
    entries: list[ColumnIndexEntry],
    hint: str,
    expected_role: ColumnRole | None,
    dataset_id: str,
    k: int = 60,
    agg_func: str = "",
) -> list[ColumnCandidate]:
    """
    Reciprocal Rank Fusion (RRF) — a fusion method that does not require weight tuning.

    RRF_score(col) = 1/(k + rank_sparse(col)) + 1/(k + rank_dense(col))
    """
    # Sparse ranking
    sparse_scored = []
    sparse_evidence_map: dict[str, list[str]] = {}
    for entry in entries:
        sc, ev = _sparse_score(entry, hint, expected_role)
        sparse_scored.append((entry.column_name, sc))
        sparse_evidence_map[entry.column_name] = ev
    sparse_scored.sort(key=lambda x: x[1], reverse=True)
    sparse_rank = {name: i + 1 for i, (name, _) in enumerate(sparse_scored)}

    # Dense ranking
    hint_type = ""
    if expected_role == ColumnRole.MEASURE:
        hint_type = "measure"
    elif expected_role == ColumnRole.DIMENSION:
        hint_type = "dimension"
    elif expected_role == ColumnRole.TIME:
        hint_type = "time"

    dense_map = _dense_scores_for_hint(hint, hint_type, dataset_id, agg_func)
    dense_sorted = sorted(dense_map.items(), key=lambda x: x[1], reverse=True)
    dense_rank = {name: i + 1 for i, (name, _) in enumerate(dense_sorted)}

    # RRF fusion
    candidates: list[ColumnCandidate] = []
    for entry in entries:
        cn = entry.column_name
        sr = sparse_rank.get(cn, len(entries))
        dr = dense_rank.get(cn, len(entries))
        rrf_score = 1.0 / (k + sr) + 1.0 / (k + dr)

        evidence = sparse_evidence_map.get(cn, []).copy()
        dn_sim = dense_map.get(cn, 0.0)
        evidence.append(f"dense_sim={dn_sim:.3f}")
        evidence.append(f"rrf(sp_rank={sr}, dn_rank={dr})")

        if rrf_score > 0:
            candidates.append(ColumnCandidate(
                dataset_id=dataset_id,
                column_name=cn,
                role=entry.inferred_role,
                score=round(rrf_score, 6),
                evidence=evidence,
                dtype=entry.dtype,
            ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:COLUMN_TOP_K]


# Stage 2 Main: Column Retrieval

def retrieve_columns(
    intent: QueryIntent,
    dataset_ids: list[str],
    fusion_method: str = "weighted",   # "weighted" | "rrf"
) -> RetrievalResult:
    """
    Stage 2: Perform hybrid column retrieval within candidate datasets.

    fusion_method:
      "weighted" — weighted linear fusion (default, tunable via RETRIEVAL_WEIGHTS)
      "rrf"      — Reciprocal Rank Fusion (no weight tuning needed)
    """
    result = RetrievalResult(dataset_candidates=dataset_ids)
    agg_func = intent.agg_func_hint.value if intent.agg_func_hint else ""

    rank_fn = _hybrid_rank_columns if fusion_method == "weighted" else _rrf_rank_columns

    for ds_id in dataset_ids:
        index_entries = load_column_index(ds_id)
        if not index_entries:
            continue

        # Metric candidates
        for hint in intent.metric_hints:
            candidates = rank_fn(
                index_entries, hint, ColumnRole.MEASURE, ds_id, agg_func=agg_func,
            )
            result.metric_candidates.extend(candidates)

        # Dimension candidates
        for hint in intent.dimension_hints:
            candidates = rank_fn(
                index_entries, hint, ColumnRole.DIMENSION, ds_id, agg_func=agg_func,
            )
            result.dimension_candidates.extend(candidates)

        # Time candidates
        for hint in intent.time_hints:
            candidates = rank_fn(
                index_entries, hint, ColumnRole.TIME, ds_id, agg_func=agg_func,
            )
            result.time_candidates.extend(candidates)

        # Filter candidates
        for fh in intent.filter_hints:
            candidates = rank_fn(
                index_entries, fh.field_hint, None, ds_id, agg_func=agg_func,
            )
            result.filter_candidates.extend(candidates)

    return result


# Public entry point

def retrieve_candidates(
    intent: QueryIntent,
    fusion_method: str = "weighted",
    domain_prior: str = None,
) -> RetrievalResult:
    """
    Main retriever entry: Hybrid Sparse-Dense Two-Stage Retrieval

    Stage 1: Dataset Retrieval  (sparse — lexical + alias + domain boost)
    Stage 2: Column Retrieval   (hybrid — sparse + dense fusion)
    """
    dataset_ids = retrieve_datasets(intent, domain_prior=domain_prior)
    result = retrieve_columns(intent, dataset_ids, fusion_method=fusion_method)

    logger.info(
        f"[Retriever] datasets={dataset_ids}, "
        f"metrics={len(result.metric_candidates)}, "
        f"dims={len(result.dimension_candidates)}, "
        f"time={len(result.time_candidates)}, "
        f"filters={len(result.filter_candidates)}"
    )

    return result

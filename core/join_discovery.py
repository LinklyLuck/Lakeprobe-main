"""
LakeProbe — Join Discovery
Function:
Find joinable columns across datasets in the data lake.
Offline: build MinHash sketches for candidate join columns (identifier/dimension)
Online:  given a column, find other columns with high value overlap
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from core.models import JoinCandidate, ColumnRole

logger = logging.getLogger(__name__)

_SKETCH_DIR = Path(__file__).parent.parent / "data" / "join_sketches"
_SKETCH_DIR.mkdir(parents=True, exist_ok=True)

MAX_SAMPLE = 50000   # max distinct values to sketch
N_HASH = 128         # number of hash functions for MinHash
LSH_BANDS = 16       # LSH bands (N_HASH / LSH_BANDS = rows per band)
LSH_ROWS = N_HASH // LSH_BANDS


# MinHash Implementation
def _hash_value(val: str, seed: int) -> int:
    """Deterministic hash with seed."""
    h = hashlib.md5(f"{seed}:{val}".encode()).hexdigest()
    return int(h[:16], 16)


def _minhash_signature(values: set[str], n_hash: int = N_HASH) -> list[int]:
    """Compute MinHash signature for a set of string values."""
    if not values:
        return [0] * n_hash
    sig = []
    for i in range(n_hash):
        min_h = min(_hash_value(str(v), i) for v in values)
        sig.append(min_h)
    return sig


def _jaccard_from_signatures(sig1: list[int], sig2: list[int]) -> float:
    """Estimate Jaccard similarity from MinHash signatures."""
    if not sig1 or not sig2 or len(sig1) != len(sig2):
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


# LSH Index
class LSHIndex:
    """Locality-Sensitive Hashing for fast candidate recall."""

    def __init__(self):
        # band_idx → {band_hash → [(dataset_id, column_name)]}
        self.buckets: list[dict[int, list[tuple[str, str]]]] = [
            {} for _ in range(LSH_BANDS)
        ]
        self.signatures: dict[tuple[str, str], list[int]] = {}
        self.value_sets: dict[tuple[str, str], set[str]] = {}

    def insert(self, dataset_id: str, column_name: str,
               signature: list[int], values: set[str]):
        key = (dataset_id, column_name)
        self.signatures[key] = signature
        self.value_sets[key] = values

        for band_idx in range(LSH_BANDS):
            start = band_idx * LSH_ROWS
            band = tuple(signature[start:start + LSH_ROWS])
            band_hash = hash(band)
            self.buckets[band_idx].setdefault(band_hash, []).append(key)

    def query(self, signature: list[int], exclude_key: tuple = None) -> set[tuple[str, str]]:
        """Find candidate keys that share at least one LSH band."""
        candidates = set()
        for band_idx in range(LSH_BANDS):
            start = band_idx * LSH_ROWS
            band = tuple(signature[start:start + LSH_ROWS])
            band_hash = hash(band)
            for key in self.buckets[band_idx].get(band_hash, []):
                if key != exclude_key:
                    candidates.add(key)
        return candidates


# Singleton LSH Index
_lsh_index: Optional[LSHIndex] = None


def get_lsh_index() -> LSHIndex:
    global _lsh_index
    if _lsh_index is None:
        _lsh_index = LSHIndex()
    return _lsh_index


def reset_lsh_index():
    global _lsh_index
    _lsh_index = None


# Offline: Build Join Sketch Index
def build_join_index_for_dataset(dataset_id: str):
    """
    Build MinHash sketches for candidate join columns in one dataset.
    Candidate columns: identifier + dimension.
    """
    from core.profiler import load_profile_card
    from config import CSV_DIR

    profile = load_profile_card(dataset_id)
    if not profile:
        return

    lsh = get_lsh_index()

    # Find CSV file
    csv_path = _find_csv(dataset_id)
    if not csv_path:
        return

    # Identify candidate join columns
    join_cols = []
    for col in profile.columns:
        if col.inferred_role in (ColumnRole.IDENTIFIER, ColumnRole.DIMENSION):
            if col.n_unique >= 2:  # at least 2 distinct values
                join_cols.append(col.name)

    if not join_cols:
        return

    # Read distinct values via DuckDB
    try:
        import duckdb
        conn = duckdb.connect()
        conn.execute(f"CREATE TABLE data AS SELECT * FROM read_csv_auto('{csv_path}')")

        for col_name in join_cols:
            try:
                rows = conn.execute(f"""
                    SELECT DISTINCT CAST("{col_name}" AS VARCHAR) as v
                    FROM data WHERE "{col_name}" IS NOT NULL
                    LIMIT {MAX_SAMPLE}
                """).fetchall()
                values = {str(r[0]) for r in rows}

                if len(values) < 2:
                    continue

                sig = _minhash_signature(values)
                lsh.insert(dataset_id, col_name, sig, values)
                logger.info(f"[JoinIndex] {dataset_id}.{col_name}: {len(values)} distinct values")

            except Exception as e:
                logger.warning(f"[JoinIndex] Skip {dataset_id}.{col_name}: {e}")

        conn.close()
    except Exception as e:
        logger.warning(f"[JoinIndex] Failed for {dataset_id}: {e}")


def build_join_index_all():
    """Build join index for all profiled datasets."""
    from core.dataset_card import load_all_dataset_cards
    reset_lsh_index()

    cards = load_all_dataset_cards()
    for card in cards:
        try:
            build_join_index_for_dataset(card.dataset_id)
        except Exception as e:
            logger.warning(f"[JoinIndex] Skip {card.dataset_id}: {e}")

    lsh = get_lsh_index()
    logger.info(f"[JoinIndex] Built index for {len(lsh.signatures)} columns")
    return len(lsh.signatures)


# Online: Find Joinable Columns
def find_joinable(
    dataset_id: str,
    column_name: str,
    top_k: int = 10,
) -> list[JoinCandidate]:
    """
    Find columns in other datasets that are joinable with the given column.
    Pipeline:
      1. Get MinHash signature for query column
      2. LSH recall: find candidate columns sharing ≥1 band
      3. Precise scoring: compute actual overlap on sampled values
      4. Rank by overlap ratio, return top-k
    """
    lsh = get_lsh_index()
    query_key = (dataset_id, column_name)

    # Check if query column is in index
    if query_key not in lsh.signatures:
        # Try to build it on the fly
        build_join_index_for_dataset(dataset_id)
        if query_key not in lsh.signatures:
            return []

    query_sig = lsh.signatures[query_key]
    query_values = lsh.value_sets[query_key]

    # LSH recall
    candidates = lsh.query(query_sig, exclude_key=query_key)

    # Precise scoring
    results = []
    for cand_key in candidates:
        cand_ds, cand_col = cand_key
        if cand_ds == dataset_id:
            continue  # skip same dataset

        cand_values = lsh.value_sets.get(cand_key, set())
        if not cand_values:
            continue

        # Compute actual overlap
        intersection = query_values & cand_values
        inter_size = len(intersection)
        min_size = min(len(query_values), len(cand_values))

        if min_size == 0 or inter_size == 0:
            continue

        overlap_ratio = inter_size / min_size
        # Also compute Jaccard for reference
        jaccard = _jaccard_from_signatures(query_sig, lsh.signatures[cand_key])

        evidence = [
            f"overlap={inter_size}/{min_size} ({overlap_ratio:.0%})",
            f"jaccard_est={jaccard:.3f}",
            f"left_distinct={len(query_values)}",
            f"right_distinct={len(cand_values)}",
        ]

        # Sample overlapping values for display
        sample_overlap = sorted(list(intersection))[:10]

        results.append(JoinCandidate(
            left_dataset=dataset_id,
            left_column=column_name,
            right_dataset=cand_ds,
            right_column=cand_col,
            overlap_ratio=round(overlap_ratio, 4),
            left_distinct=len(query_values),
            right_distinct=len(cand_values),
            intersection_size=inter_size,
            sample_overlapping_values=sample_overlap,
            evidence=evidence,
        ))

    results.sort(key=lambda r: r.overlap_ratio, reverse=True)
    return results[:top_k]


def find_all_joins_for_dataset(
    dataset_id: str,
    top_k_per_col: int = 5,
) -> dict[str, list[JoinCandidate]]:
    """Find joinable columns for all candidate columns in a dataset."""
    from core.profiler import load_profile_card

    profile = load_profile_card(dataset_id)
    if not profile:
        return {}

    results = {}
    for col in profile.columns:
        if col.inferred_role in (ColumnRole.IDENTIFIER, ColumnRole.DIMENSION):
            candidates = find_joinable(dataset_id, col.name, top_k=top_k_per_col)
            if candidates:
                results[col.name] = candidates

    return results


# Semantic Edges (from DataSearchTool FeatureGraph)
class SemanticEdge:
    """A semantic similarity edge between two columns across datasets."""
    def __init__(self, col1_ds: str, col1_name: str,
                 col2_ds: str, col2_name: str, similarity: float):
        self.col1_ds = col1_ds
        self.col1_name = col1_name
        self.col2_ds = col2_ds
        self.col2_name = col2_name
        self.similarity = similarity

    def to_dict(self):
        return {
            "left": f"{self.col1_ds}.{self.col1_name}",
            "right": f"{self.col2_ds}.{self.col2_name}",
            "similarity": round(self.similarity, 4),
        }


def find_semantically_similar_columns(
    dataset_id: str,
    column_name: str,
    top_k: int = 10,
    min_similarity: float = 0.4,
) -> list[dict]:
    """
    Find columns in OTHER datasets that are semantically similar.
    Adapted from DataSearchTool FeatureGraph.add_semantic_edge().

    Uses embedding cosine similarity (not value overlap like join discovery).
    Useful for: "this column is like that column" even if values don't overlap.
    """
    from core.embedding_engine import load_vectors, get_encoder
    from core.dataset_card import load_all_dataset_cards

    vec_data = load_vectors(dataset_id)
    if vec_data is None:
        return []

    col_names = list(vec_data["column_names"])
    if column_name not in col_names:
        return []

    idx = col_names.index(column_name)
    query_vec = vec_data["vectors"][idx]
    encoder = get_encoder()

    results = []
    for card in load_all_dataset_cards():
        if card.dataset_id == dataset_id:
            continue
        other_vec = load_vectors(card.dataset_id)
        if other_vec is None:
            continue
        sims = encoder.similarity(query_vec, other_vec["vectors"])
        for i, sim in enumerate(sims):
            if sim >= min_similarity:
                results.append({
                    "dataset": card.dataset_id,
                    "column": other_vec["column_names"][i],
                    "similarity": round(float(sim), 4),
                    "type": "semantic",
                })

    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results[:top_k]


# Utility
def _find_csv(dataset_id: str) -> Optional[Path]:
    """Find CSV file for a dataset_id."""
    from config import CSV_DIR
    # Direct match
    p = CSV_DIR / f"{dataset_id}.csv"
    if p.exists():
        return p
    # Recursive search
    candidates = list(CSV_DIR.glob(f"**/{dataset_id}.csv"))
    return candidates[0] if candidates else None


def get_join_index_stats() -> dict:
    """Get stats about the join index"""
    lsh = get_lsh_index()
    datasets = set()
    for ds, col in lsh.signatures:
        datasets.add(ds)
    return {
        "total_columns_indexed": len(lsh.signatures),
        "total_datasets": len(datasets),
        "columns": [
            {"dataset": ds, "column": col, "distinct_values": len(lsh.value_sets.get((ds, col), set()))}
            for (ds, col) in list(lsh.signatures.keys())[:50]
        ],
    }

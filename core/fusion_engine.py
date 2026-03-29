"""
LakeProbe — Evidence Fusion Engine (System Core)
Function:
  1. Candidate Merge     — Merge PartA hints + PartB candidates
  2. Hard Constraint     — Hard filtering: column existence, dtype compatibility
  3. Multi-Signal Score  — Multi-signal scoring
  4. Override Boost      — Apply user corrections (interactive refinement)
  5. Binding Selection   — Select final binding
  6. Plan Construction   — Construct ExecutablePlan
"""

from __future__ import annotations

from core.models import (
    AggFunc,
    BindingEntry,
    BindingResult,
    ColumnCandidate,
    ColumnRole,
    ExecutablePlan,
    IntentType,
    OpType,
    PlanOp,
    QueryIntent,
    RetrievalResult,
)
from config import FUSION_WEIGHTS


# 4.1 Candidate Merge
def _merge_candidates(intent: QueryIntent, candidates: RetrievalResult) -> dict:
    """
    Merge the hint from PartA and the candidate columns from PartB.
    Return {hint_type: {hint_text: [ColumnCandidate, ...]}}
    """
    merged: dict[str, dict[str, list[ColumnCandidate]]] = {
        "metric": {},
        "dimension": {},
        "time": {},
        "filter": {},
    }

    for hint in intent.metric_hints:
        matched = [c for c in candidates.metric_candidates
                   if _hint_matches_candidate(hint, c)]
        merged["metric"][hint] = matched if matched else candidates.metric_candidates[:3]

    for hint in intent.dimension_hints:
        matched = [c for c in candidates.dimension_candidates
                   if _hint_matches_candidate(hint, c)]
        merged["dimension"][hint] = matched if matched else candidates.dimension_candidates[:3]

    for hint in intent.time_hints:
        matched = [c for c in candidates.time_candidates
                   if _hint_matches_candidate(hint, c)]
        merged["time"][hint] = matched if matched else candidates.time_candidates[:3]

    for fh in intent.filter_hints:
        matched = [c for c in candidates.filter_candidates
                   if _hint_matches_candidate(fh.field_hint, c)]
        merged["filter"][fh.field_hint] = matched if matched else candidates.filter_candidates[:3]

    return merged


def _hint_matches_candidate(hint: str, cand: ColumnCandidate) -> bool:
    #Coarse match: Whether the hint is relevant to this candidate.
    h = hint.lower()
    cn = cand.column_name.lower()
    return (h in cn or cn in h or
            any(h in ev.lower() for ev in cand.evidence) or
            cand.score > 0.5)


# 4.2 Hard Constraint Filtering
_NUMERIC_DTYPES = {"int64", "float64", "int32", "float32", "number"}
_TEXT_DTYPES = {"object", "string", "str"}
_TIME_DTYPES = {"datetime64", "date"}


def _hard_filter(merged: dict, intent: QueryIntent) -> tuple[dict, list[dict]]:
    #Hard filtering is not compatible with the candidate. Returns (filtered_merged, blocked_list).
    blocked: list[dict] = []

    def filter_list(cands: list[ColumnCandidate], hint: str,
                    hint_type: str) -> list[ColumnCandidate]:
        kept = []
        for c in cands:
            reason = _check_hard_constraint(c, hint_type, intent)
            if reason:
                blocked.append({
                    "hint": hint,
                    "column": c.column_name,
                    "dataset": c.dataset_id,
                    "reason": reason,
                })
            else:
                kept.append(c)
        return kept

    for hint, cands in merged.get("metric", {}).items():
        merged["metric"][hint] = filter_list(cands, hint, "metric")

    for hint, cands in merged.get("dimension", {}).items():
        merged["dimension"][hint] = filter_list(cands, hint, "dimension")

    for hint, cands in merged.get("time", {}).items():
        merged["time"][hint] = filter_list(cands, hint, "time")

    for hint, cands in merged.get("filter", {}).items():
        merged["filter"][hint] = filter_list(cands, hint, "filter")

    return merged, blocked


def _check_hard_constraint(c: ColumnCandidate, hint_type: str,
                            intent: QueryIntent) -> str | None:
    """
    Performs a hard constraint check on a single candidate.
    Returns `None` if the check passes; otherwise, returns the reason for rejection
    """
    if hint_type == "metric":
        # COUNT / COUNT_DISTINCT
        if intent.agg_func_hint in (AggFunc.COUNT, AggFunc.COUNT_DISTINCT):
            pass  # no dtype constraint for counting
        elif c.dtype in _TEXT_DTYPES:
            return f"Cannot aggregate text column '{c.column_name}' as metric (agg={intent.agg_func_hint})"
        elif intent.agg_func_hint and c.dtype not in _NUMERIC_DTYPES:
            return f"Non-numeric column '{c.column_name}' incompatible with {intent.agg_func_hint.value}"

    elif hint_type == "dimension":
        # Dimension
        if c.role == ColumnRole.MEASURE and c.dtype in _NUMERIC_DTYPES:
            return f"Pure measure column '{c.column_name}' not suitable as dimension"

    elif hint_type == "time":
        # The time column must be of type `datetime` or contain time semantics.
        if c.dtype not in _TIME_DTYPES and c.role != ColumnRole.TIME:
            # Allow an integer column containing a year or date in its name
            if not any(kw in c.column_name.lower() for kw in ["date", "year", "month", "time", "quarter"]):
                return f"Column '{c.column_name}' does not appear to be a time column"

    return None


# 4.3 Multi-Signal Scoring
def _rescore_candidates(merged: dict) -> dict:
    """
    Re-score the candidates after hard filtering.
    Bonus magnitudes are set to meaningfully influence ranking.
    """
    for hint_type, hints_map in merged.items():
        for hint, cands in hints_map.items():
            for c in cands:
                base = c.score
                bonus = 0.0

                # Role match bonus (significant)
                if hint_type == "metric" and c.role == ColumnRole.MEASURE:
                    bonus += 0.15
                elif hint_type == "dimension" and c.role == ColumnRole.DIMENSION:
                    bonus += 0.15
                elif hint_type == "time" and c.role == ColumnRole.TIME:
                    bonus += 0.20

                # Dtype compat bonus
                if hint_type == "metric" and c.dtype in _NUMERIC_DTYPES:
                    bonus += 0.10
                elif hint_type == "time" and c.dtype in _TIME_DTYPES:
                    bonus += 0.10
                elif hint_type == "dimension" and c.dtype in _TEXT_DTYPES:
                    bonus += 0.05

                c.score = round(base + bonus, 4)

            cands.sort(key=lambda x: x.score, reverse=True)

    return merged


# 4.4 Threshold Routing + Binding Selection
def _classify_zone(score: float, max_score: float,
                   all_scores: list[float] = None) -> str:
    """
    Three-zone threshold routing with adaptive thresholds
    When score distribution is available, adapts thresholds based on
    the gap between top candidates. This handles cases where all scores
    are clustered (should be more cautious) vs clearly separated.
    """
    from config import BINDING_REJECT_THRESHOLD, BINDING_ACCEPT_THRESHOLD

    norm = score / max_score if max_score > 0 else 0

    # Adaptive: if we have the full score list, adjust thresholds
    reject_t = BINDING_REJECT_THRESHOLD
    accept_t = BINDING_ACCEPT_THRESHOLD

    if all_scores and len(all_scores) >= 2:
        sorted_scores = sorted(all_scores, reverse=True)
        top1 = sorted_scores[0]
        top2 = sorted_scores[1]
        gap = (top1 - top2) / max_score if max_score > 0 else 0

        if gap < 0.05:
            # Scores are very close — be more cautious, raise accept threshold
            accept_t = min(accept_t + 0.10, 0.90)
        elif gap > 0.30:
            # Clear winner — can be more lenient
            accept_t = max(accept_t - 0.05, 0.55)

    if norm >= accept_t:
        return "accept"
    elif norm >= reject_t:
        return "uncertain"
    else:
        return "reject"


def _select_bindings(merged: dict, dataset_ids: list[str]) -> BindingResult:
    """
     Select the final binding from the ranked candidates, with a three-zone routing label.

    Dataset selection:
      1. Retriever rank prior
      2. Column-level evidence
    """
    # Dataset selection: Combining retriever rank and column evidence
    RETRIEVER_RANK_WEIGHT = 0.4   # Weights for the Retriever prior
    COLUMN_EVIDENCE_WEIGHT = 0.6  # The weight of evidence by category

    # Retriever rank prior: The higher the rank, the higher the score
    retriever_rank_score: dict[str, float] = {}
    for i, ds_id in enumerate(dataset_ids):
        retriever_rank_score[ds_id] = 1.0 / (i + 1)

    # Column-level evidence: Calculate the average score by dataset
    dataset_col_sum: dict[str, float] = {}
    dataset_col_count: dict[str, int] = {}
    for hints_map in merged.values():
        for cands in hints_map.values():
            for c in cands:
                ds = c.dataset_id
                dataset_col_sum[ds] = dataset_col_sum.get(ds, 0) + c.score
                dataset_col_count[ds] = dataset_col_count.get(ds, 0) + 1

    dataset_col_avg: dict[str, float] = {}
    for ds in dataset_col_sum:
        dataset_col_avg[ds] = dataset_col_sum[ds] / max(dataset_col_count[ds], 1)

    # Normalize column avg to [0, 1]
    max_col_avg = max(dataset_col_avg.values()) if dataset_col_avg else 1.0
    max_rank = max(retriever_rank_score.values()) if retriever_rank_score else 1.0

    # 3) Fused score
    all_ds = set(dataset_col_avg.keys()) | set(retriever_rank_score.keys())
    dataset_fused: dict[str, float] = {}
    for ds in all_ds:
        rank_norm = retriever_rank_score.get(ds, 0) / max_rank if max_rank > 0 else 0
        col_norm = dataset_col_avg.get(ds, 0) / max_col_avg if max_col_avg > 0 else 0
        dataset_fused[ds] = RETRIEVER_RANK_WEIGHT * rank_norm + COLUMN_EVIDENCE_WEIGHT * col_norm

    best_dataset = max(dataset_fused, key=dataset_fused.get) if dataset_fused else (
        dataset_ids[0] if dataset_ids else ""
    )

    binding = BindingResult(dataset_id=best_dataset)

    # Find global max score for normalization
    all_scores = []
    for hints_map in merged.values():
        for cands in hints_map.values():
            for c in cands:
                all_scores.append(c.score)
    max_score = max(all_scores) if all_scores else 1.0

    # Select the best column and classification zone for each hint
    for hint, cands in merged.get("metric", {}).items():
        best = _pick_best(cands, best_dataset)
        cand_scores = [c.score for c in cands] if cands else []
        if best:
            zone = _classify_zone(best.score, max_score, cand_scores)
            if zone != "reject":
                best.evidence.append(f"zone={zone}")
                binding.metric_bindings.append(BindingEntry(
                    hint=hint, column=best.column_name, dataset_id=best.dataset_id,
                    score=best.score, zone=zone, evidence=best.evidence,
                ))

    for hint, cands in merged.get("dimension", {}).items():
        best = _pick_best(cands, best_dataset)
        cand_scores = [c.score for c in cands] if cands else []
        if best:
            zone = _classify_zone(best.score, max_score, cand_scores)
            if zone != "reject":
                best.evidence.append(f"zone={zone}")
                binding.dimension_bindings.append(BindingEntry(
                    hint=hint, column=best.column_name, dataset_id=best.dataset_id,
                    score=best.score, zone=zone, evidence=best.evidence,
                ))

    for hint, cands in merged.get("time", {}).items():
        best = _pick_best(cands, best_dataset)
        cand_scores = [c.score for c in cands] if cands else []
        if best:
            zone = _classify_zone(best.score, max_score, cand_scores)
            if zone != "reject":
                best.evidence.append(f"zone={zone}")
                binding.time_bindings.append(BindingEntry(
                    hint=hint, column=best.column_name, dataset_id=best.dataset_id,
                    score=best.score, zone=zone, evidence=best.evidence,
                ))

    for hint, cands in merged.get("filter", {}).items():
        best = _pick_best(cands, best_dataset)
        cand_scores = [c.score for c in cands] if cands else []
        if best:
            zone = _classify_zone(best.score, max_score, cand_scores)
            if zone != "reject":
                best.evidence.append(f"zone={zone}")
                binding.filter_bindings.append(BindingEntry(
                    hint=hint, column=best.column_name, dataset_id=best.dataset_id,
                    score=best.score, zone=zone, evidence=best.evidence,
                ))

    return binding


def _pick_best(cands: list[ColumnCandidate], preferred_dataset: str) -> ColumnCandidate | None:
    #Select the best option from the candidates, giving priority to those in preferred_dataset.
    in_ds = [c for c in cands if c.dataset_id == preferred_dataset]
    if in_ds:
        return in_ds[0]
    return cands[0] if cands else None


# 4.5 Plan Construction
def _build_plan(binding: BindingResult, intent: QueryIntent) -> ExecutablePlan:
    #Construct an operator plan based on bindings and intents, supporting automatic JOIN discovery.
    steps: list[PlanOp] = []

    # Step 1: SCAN
    steps.append(PlanOp(op=OpType.SCAN, params={"dataset": binding.dataset_id}))

    # Auto JOIN Discovery
    # If any binding columns come from a different dataset, attempt to find
    # a join path and insert JOIN ops automatically.
    cross_ds_bindings = _find_cross_dataset_bindings(binding)
    if cross_ds_bindings:
        join_ops = _build_join_ops(binding.dataset_id, cross_ds_bindings)
        steps.extend(join_ops)

    # Step 2: DERIVE_TIME
    for tb in binding.time_bindings:
        for th in intent.time_hints:
            if th.isdigit() and len(th) == 4:  # year
                steps.append(PlanOp(op=OpType.DERIVE_TIME, params={
                    "source": tb.column, "target": "derived_year", "unit": "year"
                }))
                break

    # Step 3: FILTER
    for fb in binding.filter_bindings:
        for fh in intent.filter_hints:
            if fh.field_hint.lower() in fb.hint.lower() or fb.hint.lower() in fh.field_hint.lower():
                filter_col = fb.column
                if any(s.op == OpType.DERIVE_TIME and s.params.get("source") == fb.column
                       for s in steps):
                    filter_col = "derived_year"

                steps.append(PlanOp(op=OpType.FILTER, params={
                    "column": filter_col,
                    "op": fh.op,
                    "value": fh.value,
                }))
                break

    # Step 4: GROUPBY
    group_keys = [db.column for db in binding.dimension_bindings]
    if group_keys and intent.intent_type in (IntentType.AGGREGATE, IntentType.RANKING,
                                               IntentType.TREND, IntentType.COMPARISON):
        steps.append(PlanOp(op=OpType.GROUPBY, params={"keys": group_keys}))

    # Step 5: AGGREGATE
    for mb in binding.metric_bindings:
        func = intent.agg_func_hint.value if intent.agg_func_hint else "sum"
        steps.append(PlanOp(op=OpType.AGGREGATE, params={
            "metric": mb.column,
            "func": func,
        }))

    # Step 6: SORT
    if intent.sort_hint and binding.metric_bindings:
        steps.append(PlanOp(op=OpType.SORT, params={
            "column": binding.metric_bindings[0].column,
            "order": intent.sort_hint,
        }))

    # Step 7: LIMIT
    if intent.limit_hint:
        steps.append(PlanOp(op=OpType.LIMIT, params={"n": intent.limit_hint}))

    # Step 8: SELECT
    select_cols = group_keys + [mb.column for mb in binding.metric_bindings]
    if select_cols:
        steps.append(PlanOp(op=OpType.SELECT, params={"columns": select_cols}))

    return ExecutablePlan(dataset_id=binding.dataset_id, steps=steps)


def _find_cross_dataset_bindings(binding: BindingResult) -> list[dict]:
    """
    Detect binding entries that point to columns in datasets different
    from the primary dataset. These indicate a potential JOIN need.
    """
    primary = binding.dataset_id
    cross = []
    for group_name, entries in [
        ("metric", binding.metric_bindings),
        ("dimension", binding.dimension_bindings),
        ("time", binding.time_bindings),
        ("filter", binding.filter_bindings),
    ]:
        for entry in entries:
            if entry.dataset_id and entry.dataset_id != primary:
                cross.append({
                    "type": group_name,
                    "hint": entry.hint,
                    "column": entry.column,
                    "target_dataset": entry.dataset_id,
                })
    return cross


def _build_join_ops(primary_dataset: str, cross_bindings: list[dict]) -> list[PlanOp]:
    """
    For each cross-dataset binding, attempt to find a joinable column pair
    using the MinHash join index. Returns JOIN PlanOps.
    """
    join_ops = []
    joined_datasets = set()

    for cb in cross_bindings:
        target_ds = cb["target_dataset"]
        if target_ds in joined_datasets:
            continue  # already joined this dataset

        # Try to find a join key between primary and target
        join_key = _find_join_key(primary_dataset, target_ds)
        if join_key:
            join_ops.append(PlanOp(op=OpType.JOIN, params={
                "right_dataset": target_ds,
                "left_key": join_key[0],
                "right_key": join_key[1],
                "join_type": "LEFT",
            }))
            joined_datasets.add(target_ds)

    return join_ops


def _find_join_key(left_ds: str, right_ds: str) -> tuple[str, str] | None:
    """
    Find the best join key pair between two datasets using join discovery.
    Returns (left_column, right_column) or None.
    """
    try:
        from core.join_discovery import find_all_joins_for_dataset
        all_joins = find_all_joins_for_dataset(left_ds, top_k_per_col=3)
        best_overlap = 0.0
        best_pair = None

        for left_col, candidates in all_joins.items():
            for jc in candidates:
                if jc.right_dataset == right_ds and jc.overlap_ratio > best_overlap:
                    best_overlap = jc.overlap_ratio
                    best_pair = (left_col, jc.right_column)

        # Only accept joins with >= 50% value overlap
        if best_pair and best_overlap >= 0.5:
            return best_pair
    except Exception:
        pass

    return None


# External main function
def fuse_and_plan(
    intent: QueryIntent,
    candidates: RetrievalResult,
) -> tuple[BindingResult, ExecutablePlan, dict]:
    """
    Evidence Fusion Engine -main。

    Pipeline: merge → hard filter → rescore → override boost → bind → plan

    Returns: (binding, plan, override_info)
    """
    from core.override_store import (
        get_override_store,
        apply_overrides_to_candidates,
    )

    # 1. Merge
    merged = _merge_candidates(intent, candidates)

    # 2. Hard Constraint Filtering
    merged, blocked = _hard_filter(merged, intent)

    # 3. Multi-Signal Scoring
    merged = _rescore_candidates(merged)
    """
        1.Override Boost (interactive refinement)
        2.After the hard filter and before binding selection
        3.Ensure that overrides do not bypass physical constraints
    """
    store = get_override_store()

    # identify the candidate dataset
    dataset_scores: dict[str, float] = {}
    for hints_map in merged.values():
        for cands in hints_map.values():
            for c in cands:
                dataset_scores[c.dataset_id] = dataset_scores.get(c.dataset_id, 0) + c.score
    likely_dataset = max(dataset_scores, key=dataset_scores.get) if dataset_scores else (
        candidates.dataset_candidates[0] if candidates.dataset_candidates else ""
    )

    merged, override_result = apply_overrides_to_candidates(
        merged, likely_dataset, store,
    )
    override_info = override_result.model_dump()

    # 5. Binding Selection
    binding = _select_bindings(merged, candidates.dataset_candidates)
    binding.blocked_candidates = blocked

    # 6. Plan Construction
    plan = _build_plan(binding, intent)

    return binding, plan, override_info

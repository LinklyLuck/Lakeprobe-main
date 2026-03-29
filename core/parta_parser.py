"""
LakeProbe — PartA: Query → Schema-Agnostic Intent

Function:
DOMAIN-AGNOSTIC: No hardcoded column names or domain-specific keywords.
Works with wine, housing, IoT, medical, finance, or any CSV data lake.
Four sub-moudles:
  1. Query Parser        — NL → QueryIntent (LLM preferred, smart fallback)
  2. Semantic Validator   — check intent completeness (domain-agnostic rules)
  3. Ambiguity Detector   — flag ambiguities for user review
  4. Intent Normalizer    — synonym normalization via knowledge base
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from core.models import (
    AggFunc,
    FilterHint,
    IntentType,
    QueryIntent,
)


# Query Parser (LLM preferred)
PARSE_SYSTEM_PROMPT = """\
You are a query intent parser for a data lake. Given a natural language question,
extract a STRUCTURED intent. Do NOT guess column names — use semantic hints only.

Return STRICTLY valid JSON (no markdown fences) with these fields:
{
  "intent_type": "aggregate|trend|ranking|filter|comparison|distribution|correlation|lookup|unknown",
  "metric_hints": ["<semantic keywords for measures — whatever the user wants to measure>"],
  "dimension_hints": ["<semantic keywords for grouping — whatever the user wants to group by>"],
  "filter_hints": [{"field_hint": "<keyword>", "op": "=|>|<|>=|<=|!=|in|between", "value": "<val>"}],
  "time_hints": ["<time references>"],
  "sort_hint": "asc|desc|null",
  "limit_hint": <int or null>,
  "agg_func_hint": "sum|avg|count|min|max|median|count_distinct|null",
  "confidence": "high|medium|low"
}

RULES:
1. Extract the actual domain terms from the query — do NOT invent or normalize them.
2. Keep multi-word concepts together: "bond type", "lap time", "blood pressure" → single hint.
3. Words like "type", "level", "value", "status", "name", "data" CAN be part of domain hints — do NOT strip them.
4. If the query is ambiguous or you are unsure about the intent, set confidence to "low" and intent_type to "unknown".
5. For Chinese/non-English queries, extract the semantic concepts in the original language.

Examples across domains AND sentence structures:

Q: "top 5 wines by quality"
A: {"intent_type":"ranking","metric_hints":["quality"],"dimension_hints":["wine"],"filter_hints":[],"time_hints":[],"sort_hint":"desc","limit_hint":5,"agg_func_hint":"max","confidence":"high"}

Q: "average alcohol by quality level"
A: {"intent_type":"aggregate","metric_hints":["alcohol"],"dimension_hints":["quality level"],"filter_hints":[],"time_hints":[],"sort_hint":null,"limit_hint":null,"agg_func_hint":"avg","confidence":"high"}

Q: "compare male and female cases"
A: {"intent_type":"comparison","metric_hints":["cases"],"dimension_hints":["gender"],"filter_hints":[],"time_hints":[],"sort_hint":null,"limit_hint":null,"agg_func_hint":"sum","confidence":"high"}

Q: "total goals by league in 2016 season"
A: {"intent_type":"aggregate","metric_hints":["goals"],"dimension_hints":["league"],"filter_hints":[{"field_hint":"season","op":"=","value":"2016"}],"time_hints":[],"sort_hint":"desc","limit_hint":null,"agg_func_hint":"sum","confidence":"high"}

Q: "most common bond type in molecules"
A: {"intent_type":"aggregate","metric_hints":["bond type"],"dimension_hints":[],"filter_hints":[],"time_hints":[],"sort_hint":"desc","limit_hint":1,"agg_func_hint":"count","confidence":"high"}

Q: "which customer had the least consumption"
A: {"intent_type":"ranking","metric_hints":["consumption"],"dimension_hints":["customer"],"filter_hints":[],"time_hints":[],"sort_hint":"asc","limit_hint":1,"agg_func_hint":"min","confidence":"high"}

Q: "what percentage of orders were returned last quarter"
A: {"intent_type":"aggregate","metric_hints":["orders","returned"],"dimension_hints":[],"filter_hints":[],"time_hints":["last quarter"],"sort_hint":null,"limit_hint":null,"agg_func_hint":"count","confidence":"medium"}

Q: "show me all transactions where amount exceeds 10000"
A: {"intent_type":"filter","metric_hints":["amount"],"dimension_hints":[],"filter_hints":[{"field_hint":"amount","op":">","value":10000}],"time_hints":[],"sort_hint":null,"limit_hint":null,"agg_func_hint":null,"confidence":"high"}

Q: "how has monthly revenue changed over the past year"
A: {"intent_type":"trend","metric_hints":["revenue"],"dimension_hints":[],"filter_hints":[],"time_hints":["monthly","past year"],"sort_hint":null,"limit_hint":null,"agg_func_hint":"sum","confidence":"high"}

Q: "distribution of patient ages in the ICU"
A: {"intent_type":"distribution","metric_hints":["age"],"dimension_hints":[],"filter_hints":[{"field_hint":"department","op":"=","value":"ICU"}],"time_hints":[],"sort_hint":null,"limit_hint":null,"agg_func_hint":null,"confidence":"high"}

Q: "list all products that have never been ordered"
A: {"intent_type":"filter","metric_hints":["products","orders"],"dimension_hints":[],"filter_hints":[{"field_hint":"order count","op":"=","value":0}],"time_hints":[],"sort_hint":null,"limit_hint":null,"agg_func_hint":null,"confidence":"medium"}

Q: "average enrollment by school county"
A: {"intent_type":"aggregate","metric_hints":["enrollment"],"dimension_hints":["county"],"filter_hints":[],"time_hints":[],"sort_hint":null,"limit_hint":null,"agg_func_hint":"avg","confidence":"high"}

Q: "correlation between temperature and energy usage"
A: {"intent_type":"correlation","metric_hints":["temperature","energy usage"],"dimension_hints":[],"filter_hints":[],"time_hints":[],"sort_hint":null,"limit_hint":null,"agg_func_hint":null,"confidence":"high"}

Q: "按地区统计销售额"
A: {"intent_type":"aggregate","metric_hints":["销售额"],"dimension_hints":["地区"],"filter_hints":[],"time_hints":[],"sort_hint":null,"limit_hint":null,"agg_func_hint":"sum","confidence":"high"}

Q: "找出退货率最高的前10个供应商"
A: {"intent_type":"ranking","metric_hints":["退货率"],"dimension_hints":["供应商"],"filter_hints":[],"time_hints":[],"sort_hint":"desc","limit_hint":10,"agg_func_hint":"max","confidence":"high"}

Q: "what are the best performing regions"
A: {"intent_type":"ranking","metric_hints":[],"dimension_hints":["region"],"filter_hints":[],"time_hints":[],"sort_hint":"desc","limit_hint":null,"agg_func_hint":null,"confidence":"low"}
"""

# Stop words for keyword extraction (domain-agnostic)
_STOP_WORDS = frozenset({
    "i", "me", "my", "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "am", "do", "does", "did", "will", "would", "could", "should", "shall", "may",
    "might", "can", "have", "has", "had", "having", "to", "of", "in", "for", "on",
    "at", "from", "by", "with", "as", "and", "or", "but", "if", "than", "that",
    "this", "these", "those", "it", "its", "what", "which", "who", "whom", "how",
    "when", "where", "why", "not", "no", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "only", "own", "same", "so", "too",
    "very", "just", "about", "above", "after", "before", "between", "into", "out",
    "up", "down", "over", "under", "again", "further", "then", "once", "also",
    # Query structure words (NOT domain words)
    "show", "me", "give", "get", "find", "list", "display", "tell", "what",
    "how", "much", "many", "there", "please", "want", "need", "looking",
})

# Aggregation keywords (domain-agnostic)
_AGG_KEYWORDS = {
    "sum": ["sum", "total", "总和", "合计", "总额", "总"],
    "avg": ["average", "avg", "mean", "平均"],
    "count": ["count", "number", "数量", "多少个", "how many"],
    "max": ["max", "maximum", "highest", "最大", "最高"],
    "min": ["min", "minimum", "lowest", "最小", "最低"],
    "median": ["median", "中位数"],
}

# Preposition/structure words that indicate "by X" grouping.
_BY_WORDS = frozenset({"by", "per", "across", "for each", "grouped by", "分", "按", "每个", "各"})

# Filter operator patterns
_FILTER_PATTERNS = [
    (r"(?:greater than|more than|above|over|>)\s*(\d+\.?\d*)", ">"),
    (r"(?:less than|below|under|<)\s*(\d+\.?\d*)", "<"),
    (r"(?:equal to|equals|=)\s*(\w+)", "="),
    (r"(\w+)\s*=\s*(\d+)", "="),
]


def _call_llm(query: str) -> dict:
    """Call LLM for intent parsing. Falls back to smart keyword parser."""
    try:
        import openai
        from config import LLM_MODEL, LLM_TEMPERATURE, LLM_API_KEY, LLM_API_BASE

        # Read key from config FIRST, then env vars
        api_key = LLM_API_KEY or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        api_base = LLM_API_BASE or os.getenv("LLM_API_BASE")

        if not api_key:
            try:
                from core.plan_optimizer import measure_lakeprobe_tokens_tiktoken
                measure_lakeprobe_tokens_tiktoken(PARSE_SYSTEM_PROMPT, query)
            except Exception:
                pass
            return _fallback_rule_parse(query)

        client_kwargs: dict = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base

        client = openai.OpenAI(**client_kwargs, timeout=60)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": PARSE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )

        try:
            from core.plan_optimizer import record_llm_tokens
            usage = resp.usage
            if usage:
                record_llm_tokens(usage.prompt_tokens, usage.completion_tokens)
        except Exception:
            pass

        text = resp.choices[0].message.content.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)

    except Exception:
        try:
            from core.plan_optimizer import measure_lakeprobe_tokens_tiktoken
            measure_lakeprobe_tokens_tiktoken(PARSE_SYSTEM_PROMPT, query)
        except Exception:
            pass
        return _fallback_rule_parse(query)


def _fallback_rule_parse(query: str) -> dict:
    """
    Domain-agnostic fallback parser with multi-signal intent detection.

    Improvements over original:
    1.Multi-signal scoring for intent type (not first-match-wins)
    2.Unicode-aware tokenization (supports Chinese, Japanese, etc.)
    3.Handles multiple "by" clauses and complex sentence structures
    4.Does NOT strip domain-meaningful words like "type", "level", "value"
    """
    q = query.lower().strip()
    result: dict = {
        "intent_type": "unknown",
        "metric_hints": [],
        "dimension_hints": [],
        "filter_hints": [],
        "time_hints": [],
        "sort_hint": None,
        "limit_hint": None,
        "agg_func_hint": None,
        "confidence": "medium",
    }

    #Intent type detection (multi-signal scoring)
    intent_scores: dict[str, float] = {
        "ranking": 0, "trend": 0, "comparison": 0, "distribution": 0,
        "correlation": 0, "filter": 0, "aggregate": 0, "lookup": 0,
    }
    ranking_kw = ["top", "rank", "best", "worst", "highest", "lowest",
                  "最高", "最低", "前", "最大", "最小", "most", "least", "biggest"]
    trend_kw = ["trend", "over time", "change", "变化", "趋势",
                "monthly", "yearly", "daily", "weekly", "growth", "decline"]
    comparison_kw = ["compare", "versus", "vs", "对比", "比较",
                     "difference between", "differ"]
    distribution_kw = ["distribution", "分布", "histogram", "spread", "breakdown"]
    correlation_kw = ["correlation", "relationship", "相关", "correlate"]
    filter_kw = ["filter", "where", "greater than", "less than", "above",
                 "below", "exceeds", "筛选", "过滤"]
    aggregate_kw = ["total", "sum", "average", "avg", "count", "aggregate",
                    "总", "合计", "平均", "数量", "统计", "多少"]
    lookup_kw = ["show me", "list", "find", "get", "display", "查看", "列出", "找"]

    for kw in ranking_kw:
        if kw in q: intent_scores["ranking"] += 2.0
    for kw in trend_kw:
        if kw in q: intent_scores["trend"] += 2.0
    for kw in comparison_kw:
        if kw in q: intent_scores["comparison"] += 2.0
    for kw in distribution_kw:
        if kw in q: intent_scores["distribution"] += 2.0
    for kw in correlation_kw:
        if kw in q: intent_scores["correlation"] += 2.0
    for kw in filter_kw:
        if kw in q: intent_scores["filter"] += 1.5
    for kw in aggregate_kw:
        if kw in q: intent_scores["aggregate"] += 1.5
    for kw in lookup_kw:
        if kw in q: intent_scores["lookup"] += 1.0

    best_intent = max(intent_scores, key=intent_scores.get)
    best_score = intent_scores[best_intent]
    if best_score > 0:
        result["intent_type"] = best_intent
        second_score = sorted(intent_scores.values(), reverse=True)[1]
        if best_score - second_score < 1.0:
            result["confidence"] = "low"
    else:
        result["intent_type"] = "unknown"
        result["confidence"] = "low"

    #Agg function
    for func, keywords in _AGG_KEYWORDS.items():
        if any(k in q for k in keywords):
            result["agg_func_hint"] = func
            break

    #Extract content words
    # Match English words, underscored compounds, AND CJK characters/phrases
    en_tokens = re.findall(r'[a-zA-Z_]{2,}', q)
    # CJK token extraction (simple: consecutive CJK chars as one token)
    cjk_tokens = re.findall(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]{2,}', query)

    expanded = []
    for t in en_tokens:
        expanded.append(t)
        if "_" in t:
            expanded.extend(p for p in t.split("_") if len(p) > 1)

    # Structure words that should be removed
    agg_words = set()
    for kws in _AGG_KEYWORDS.values():
        agg_words.update(k.lower() for k in kws if " " not in k)
    structure_words = _STOP_WORDS | agg_words | {
        "top", "rank", "compare", "distribution", "correlation",
        "trend", "filter", "show", "between", "across",
        "group", "grouped",
    }
    # NOTE: "type", "level", "value", "data", "name", "status" are NOT stripped

    content_words = [t for t in expanded if t.lower() not in structure_words]

    #Chinese structural word filtering
    cn_struct = {"的", "是", "在", "和", "与", "或", "了", "每个", "所有", "哪个", "什么", "怎么",
                 "请", "帮我", "告诉", "显示", "查看", "列出", "找出"}
    cn_content = [t for t in cjk_tokens if t not in cn_struct]

    #Split metric vs dimension
    # Find ALL "by"/"per"/"按" positions, use the LAST one for splitting
    q_words = q.split()
    by_positions = [i for i, w in enumerate(q_words)
                    if w in ("by", "per", "across", "grouped")]

    # Chinese
    cn_by_match = re.search(r'按(.+?)(?:统计|计算|分析|的|$)', q)
    cn_mei_match = re.search(r'每(?:个)?(.+?)(?:的|统计|$)', q)

    if by_positions:
        by_pos = by_positions[-1]  # Use LAST "by" position
        before_by = " ".join(q_words[:by_pos])
        after_by = " ".join(q_words[by_pos + 1:])

        before_tokens = [t for t in re.findall(r'[a-zA-Z_]{2,}', before_by)
                         if t.lower() not in structure_words]
        after_tokens = [t for t in re.findall(r'[a-zA-Z_]{2,}', after_by)
                        if t.lower() not in structure_words]

        if result["intent_type"] == "ranking":
            result["metric_hints"] = after_tokens[:3] if after_tokens else before_tokens[:2]
            result["dimension_hints"] = before_tokens[:2] if after_tokens else []
        else:
            result["metric_hints"] = before_tokens[:3] if before_tokens else []
            result["dimension_hints"] = after_tokens[:3] if after_tokens else []

    elif cn_by_match or cn_mei_match:
        # Chinese: "按地区统计销售额" → dim=地区, metric=销售额
        if cn_by_match:
            dim_part = cn_by_match.group(1).strip()
            result["dimension_hints"] = [dim_part] if dim_part else []
            remaining = [t for t in cn_content if t != dim_part]
            result["metric_hints"] = remaining[:2]
        elif cn_mei_match:
            dim_part = cn_mei_match.group(1).strip()
            result["dimension_hints"] = [dim_part] if dim_part else []
            remaining = [t for t in cn_content if t != dim_part]
            result["metric_hints"] = remaining[:2]

    elif result["intent_type"] == "correlation":
        and_pos = q.find(" and ")
        if and_pos > 0 and "between" in q:
            between_pos = q.find("between")
            segment = q[between_pos + 7:].strip()
            parts = segment.split(" and ")
            corr_tokens = []
            for part in parts:
                words = [t for t in re.findall(r'[a-zA-Z_]{2,}', part)
                         if t.lower() not in structure_words]
                corr_tokens.extend(words[:2])
            result["metric_hints"] = corr_tokens[:2]
        else:
            result["metric_hints"] = content_words[:2]

    elif result["intent_type"] == "comparison":
        result["metric_hints"] = content_words[-1:] if content_words else []
        result["dimension_hints"] = content_words[:-1][:2] if len(content_words) > 1 else []

    else:
        # Fallback: check for "of" pattern
        of_pos = -1
        for i, w in enumerate(q_words):
            if w == "of":
                of_pos = i
                break
        if of_pos >= 0:
            after_of = " ".join(q_words[of_pos + 1:])
            of_tokens = [t for t in re.findall(r'[a-zA-Z_]{2,}', after_of)
                         if t.lower() not in structure_words]
            result["metric_hints"] = of_tokens[:3] if of_tokens else []
        else:
            all_hints = content_words + cn_content
            result["metric_hints"] = all_hints[:2] if all_hints else []
            result["dimension_hints"] = all_hints[2:4] if len(all_hints) > 2 else []

    # Add CJK content to hints if not already populated
    if not result["metric_hints"] and cn_content:
        result["metric_hints"] = cn_content[:2]
    if not result["dimension_hints"] and len(cn_content) > 2:
        result["dimension_hints"] = cn_content[2:4]

    #Time hints
    year_matches = re.findall(r"20\d{2}", query)
    result["time_hints"].extend(year_matches)
    for kw in ["last year", "去年", "今年", "this year", "last quarter",
               "last month", "上个月", "上季度", "monthly", "yearly", "daily",
               "weekly", "over time", "past"]:
        if kw in q:
            result["time_hints"].append(kw)

    #Sort / Limit
    top_match = re.search(r"top\s*(\d+)", q)
    if top_match:
        result["limit_hint"] = int(top_match.group(1))
        result["sort_hint"] = "desc"
    qian_match = re.search(r"前\s*(\d+)", q)
    if qian_match:
        result["limit_hint"] = int(qian_match.group(1))
        result["sort_hint"] = "desc"

    #Filter hints
    for pattern, op in _FILTER_PATTERNS:
        m = re.search(pattern, q)
        if m:
            val = m.group(1)
            try:
                val = float(val) if "." in val else int(val)
            except ValueError:
                pass
            preceding = q[:m.start()].strip().split()
            field = preceding[-1] if preceding else "value"
            if field.lower() in _STOP_WORDS:
                field = result["metric_hints"][0] if result["metric_hints"] else "value"
            result["filter_hints"].append({
                "field_hint": field, "op": op, "value": val
            })
            break

    for y in year_matches:
        result["filter_hints"].append({"field_hint": "year", "op": "=", "value": int(y)})

    #Knowledge base enrichment
    try:
        from core.knowledge_base import expand_abbreviation
        enriched_metrics = []
        for m in result["metric_hints"]:
            enriched_metrics.append(m)
            exp = expand_abbreviation(m)
            if exp:
                enriched_metrics.extend(exp[:1])
        result["metric_hints"] = list(dict.fromkeys(enriched_metrics))[:4]

        enriched_dims = []
        for d in result["dimension_hints"]:
            enriched_dims.append(d)
            exp = expand_abbreviation(d)
            if exp:
                enriched_dims.extend(exp[:1])
        result["dimension_hints"] = list(dict.fromkeys(enriched_dims))[:4]
    except Exception:
        pass

    return result


def _dict_to_query_intent(raw: dict, query: str) -> QueryIntent:
    """Convert parser dict to QueryIntent."""
    intent_type = IntentType.UNKNOWN
    try:
        intent_type = IntentType(raw.get("intent_type", "unknown"))
    except ValueError:
        pass

    agg_func = None
    if raw.get("agg_func_hint"):
        try:
            agg_func = AggFunc(raw["agg_func_hint"])
        except ValueError:
            pass

    filter_hints = []
    for fh in raw.get("filter_hints", []):
        if isinstance(fh, dict):
            filter_hints.append(FilterHint(**fh))

    return QueryIntent(
        intent_type=intent_type,
        metric_hints=raw.get("metric_hints", []),
        dimension_hints=raw.get("dimension_hints", []),
        filter_hints=filter_hints,
        time_hints=[str(t) for t in raw.get("time_hints", [])],
        sort_hint=raw.get("sort_hint"),
        limit_hint=raw.get("limit_hint"),
        agg_func_hint=agg_func,
        raw_query=query,
    )


#Semantic Validator (domain-agnostic rules)
def _validate_intent(intent: QueryIntent) -> list[str]:
    """Check intent completeness. Rules are structural, not domain-specific."""
    warnings: list[str] = []

    if intent.intent_type == IntentType.AGGREGATE and not intent.metric_hints:
        warnings.append("Aggregate query but no metric hint detected.")

    if intent.intent_type == IntentType.TREND and not intent.time_hints:
        warnings.append("Trend query but no time reference detected.")

    if intent.intent_type == IntentType.RANKING:
        if not intent.metric_hints:
            warnings.append("Ranking query but no metric to rank by.")
        if intent.sort_hint is None:
            intent.sort_hint = "desc"
        if intent.limit_hint is None:
            warnings.append("Ranking query, defaulting to top 10.")
            intent.limit_hint = 10

    if intent.intent_type == IntentType.COMPARISON and not intent.dimension_hints:
        warnings.append("Comparison query but no dimension to compare.")

    return warnings


# Ambiguity Detector (domain-agnostic)
def _detect_ambiguities(intent: QueryIntent) -> list[str]:
    """Flag ambiguities for user review. No domain-specific suggestions."""
    ambiguities: list[str] = []

    if intent.intent_type in (IntentType.AGGREGATE, IntentType.RANKING,
                               IntentType.TREND) and not intent.metric_hints:
        ambiguities.append(
            "No metric detected. Please specify what to measure "
            "(e.g., the column name or concept you want to aggregate)."
        )

    q = intent.raw_query.lower()
    if any(k in q for k in ["best", "最好", "最优"]) and not intent.metric_hints:
        ambiguities.append(
            '"Best" is ambiguous — please specify which measure to rank by.'
        )

    if any(k in q for k in ["recent", "最近", "lately"]) and not intent.time_hints:
        ambiguities.append(
            '"Recent" is ambiguous — please specify a time range.'
        )

    return ambiguities


# Intent Normalizer (knowledge base)
def _normalize_hints(hints: list[str]) -> list[str]:
    """Normalize hints via knowledge base canonical forms."""
    normalized: list[str] = []
    for h in hints:
        key = h.lower().strip()
        try:
            from core.knowledge_base import get_canonical
            canonical = get_canonical(key)
        except Exception:
            canonical = key
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def _normalize_intent(intent: QueryIntent) -> QueryIntent:
    """Normalize all hints in intent."""
    intent.metric_hints = _normalize_hints(intent.metric_hints)
    intent.dimension_hints = _normalize_hints(intent.dimension_hints)
    for fh in intent.filter_hints:
        try:
            from core.knowledge_base import get_canonical
            fh.field_hint = get_canonical(fh.field_hint.lower())
        except Exception:
            pass
    return intent


# Main entry
def build_query_intent(query: str) -> QueryIntent:
    """
    PartA main entry: NL → QueryIntent

    Pipeline: parse → validate → detect ambiguity → normalize
    """
    raw = _call_llm(query)
    intent = _dict_to_query_intent(raw, query)

    warnings = _validate_intent(intent)
    ambiguities = _detect_ambiguities(intent)
    intent.ambiguities = ambiguities
    intent.needs_clarification = len(ambiguities) > 0

    intent = _normalize_intent(intent)
    return intent
"""
LakeProbe — LLM-Based Alias Generator
Function:
Replaces hardcoded ALIAS_LEXICON with dynamic LLM-generated aliases.
Flow:
  1. During column indexing, send all column names + sample values to LLM
  2. LLM returns aliases/synonyms for each column
  3. Cache results to disk (per dataset) so we only call LLM once
  4. Fallback to ALIAS_LEXICON if LLM unavailable
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Cache directory
_ALIAS_CACHE_DIR = Path(__file__).parent.parent / "data" / "alias_cache"
_ALIAS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# LLM Alias Generation Prompt
_ALIAS_SYSTEM_PROMPT = """\
You are a data column alias generator. Given a list of column names from a dataset,
generate synonyms and alternative names that a user might use in natural language queries.

For each column, provide 3-8 aliases that capture:
- Common synonyms (e.g., "revenue" for "sales_amount")
- Abbreviations (e.g., "qty" for "quantity")
- Natural language forms (e.g., "how much" for "price")
- Domain-specific terms (e.g., "ABV" for "alcohol")
- Related concepts (e.g., "sweetness" for "residual_sugar")

Return STRICTLY valid JSON (no markdown fences):
{
  "column_name_1": ["alias1", "alias2", "alias3"],
  "column_name_2": ["alias1", "alias2"]
}

Only return the JSON object. No explanation."""


def _build_alias_prompt(columns: list[dict]) -> str:
    """Build the user prompt with column info."""
    lines = ["Dataset columns:\n"]
    for col in columns:
        parts = [f"- {col['name']} (type={col['dtype']}, role={col['role']})"]
        if col.get("samples"):
            parts.append(f"  samples: {', '.join(str(s) for s in col['samples'][:5])}")
        lines.append("\n".join(parts))
    return "\n".join(lines)


# LLM Call
def _call_llm_for_aliases(columns: list[dict]) -> Optional[dict[str, list[str]]]:
    """Call LLM to generate aliases for columns. Returns {col_name: [aliases]} or None."""
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
        user_prompt = _build_alias_prompt(columns)

        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": _ALIAS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        text = resp.choices[0].message.content.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        result = json.loads(text)

        # Record token usage
        try:
            from core.plan_optimizer import record_llm_tokens
            usage = resp.usage
            if usage:
                record_llm_tokens(usage.prompt_tokens, usage.completion_tokens)
        except Exception:
            pass

        logger.info(f"[LLM Alias] Generated aliases for {len(result)} columns")
        return result

    except Exception as e:
        logger.warning(f"[LLM Alias] Failed: {e}")
        return None


# Cache Management
def _cache_path(dataset_id: str) -> Path:
    return _ALIAS_CACHE_DIR / f"{dataset_id}_aliases.json"


def _load_cached_aliases(dataset_id: str) -> Optional[dict[str, list[str]]]:
    path = _cache_path(dataset_id)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            logger.info(f"[LLM Alias] Cache hit for '{dataset_id}' ({len(data)} columns)")
            return data
        except Exception:
            pass
    return None


def _save_aliases_cache(dataset_id: str, aliases: dict[str, list[str]]):
    path = _cache_path(dataset_id)
    path.write_text(json.dumps(aliases, indent=2, ensure_ascii=False), encoding="utf-8")



# Main Entry Point
def generate_aliases_for_dataset(
    dataset_id: str,
    columns: list[dict],
    force_refresh: bool = False,
) -> dict[str, list[str]]:
    """
    Generate aliases for all columns in a dataset.

    Priority (each layer adds to the previous):
      1. Cache hit → return cached
      2. WordNet + Extended Dict (free, offline, fast) → base aliases
      3. LLM (API, richer, domain-specific) → enrich on top
      4. ALIAS_LEXICON (hardcoded, last resort)

    All layers merge — WordNet provides linguistic synonyms,
    LLM adds domain-specific terms, ALIAS_LEXICON covers gaps.
    """
    # 1. Check cache
    if not force_refresh:
        cached = _load_cached_aliases(dataset_id)
        if cached:
            return cached

    merged = {}
    for col_info in columns:
        name = col_info["name"]
        base = name.lower().strip()
        aliases = {base, base.replace("_", " "), base.replace("_", "")}

        # 2. WordNet + Extended Dict + Abbreviation (free, offline)
        try:
            from core.knowledge_base import get_synonyms, get_canonical, expand_abbreviation
            from config import KB_USE_WORDNET, KB_USE_CONCEPTNET

            kb_syns = get_synonyms(
                base,
                use_wordnet=KB_USE_WORDNET,
                use_conceptnet=KB_USE_CONCEPTNET,
            )
            aliases.update(kb_syns)

            # Canonical form
            canon = get_canonical(base)
            if canon != base:
                aliases.add(canon)

            # Abbreviation expansion (e.g., "qty" → "quantity", "cust_seg" → "customer segment")
            abbr_exp = expand_abbreviation(base)
            aliases.update(abbr_exp)

            # Multi-word: split and look up each token
            tokens = base.replace("_", " ").replace("-", " ").split()
            for t in tokens:
                if len(t) > 2:
                    t_syns = get_synonyms(t, use_wordnet=KB_USE_WORDNET, use_conceptnet=False)
                    aliases.update(t_syns[:3])
                    t_canon = get_canonical(t)
                    if t_canon != t:
                        aliases.add(t_canon)
                # Expand abbreviations for each token too
                t_exp = expand_abbreviation(t)
                aliases.update(t_exp[:2])

        except Exception as e:
            logger.debug(f"[Alias] Knowledge base failed for '{name}': {e}")

        merged[name] = aliases  # store as set for now

    # 3. Try LLM (enrich on top of WordNet results)
    llm_result = _call_llm_for_aliases(columns)
    if llm_result:
        for col_info in columns:
            name = col_info["name"]
            llm_aliases = {a.lower().strip() for a in llm_result.get(name, [])}
            if name in merged:
                merged[name] = merged[name] | llm_aliases
            else:
                merged[name] = llm_aliases
    else:
        # 4. If no LLM, ensure ALIAS_LEXICON fallback for missing ones
        from config import ALIAS_LEXICON, ALIAS_REVERSE
        for col_info in columns:
            name = col_info["name"]
            base = name.lower().strip()
            if name not in merged:
                merged[name] = {base}

            canonical = ALIAS_REVERSE.get(base)
            if canonical:
                merged[name].add(canonical)
                for a in ALIAS_LEXICON.get(canonical, []):
                    merged[name].add(a.lower())

            tokens = base.replace("_", " ").replace("-", " ").split()
            for t in tokens:
                canon = ALIAS_REVERSE.get(t)
                if canon:
                    merged[name].add(canon)

    # Convert sets to sorted lists, remove empty
    final = {}
    for name, alias_set in merged.items():
        final[name] = sorted(alias_set - {""})

    # Cache result
    _save_aliases_cache(dataset_id, final)
    return final


def _fallback_aliases(columns: list[dict]) -> dict[str, list[str]]:
    """Ultimate fallback: just ALIAS_LEXICON (no WordNet, no LLM)."""
    from config import ALIAS_LEXICON, ALIAS_REVERSE

    result = {}
    for col_info in columns:
        name = col_info["name"]
        base = name.lower().strip()
        aliases = {base, base.replace("_", " "), base.replace("_", "")}

        canonical = ALIAS_REVERSE.get(base)
        if canonical:
            aliases.add(canonical)
            for a in ALIAS_LEXICON.get(canonical, []):
                aliases.add(a.lower())

        tokens = base.replace("_", " ").replace("-", " ").split()
        for t in tokens:
            canon = ALIAS_REVERSE.get(t)
            if canon:
                aliases.add(canon)

        result[name] = sorted(aliases - {""})

    return result

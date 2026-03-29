"""
LakeProbe — Domain Router
Fuction:
IR-based Domain Routing for Schema Grounding.
Position in pipeline:
  PartA (structural IR) → Domain Router → Retriever (boosted) → Fusion → Plan
The router infers the most likely domain of a query from its structured IR,
producing a soft routing prior — NOT a hard filter.
Three modes (for ablation):
  1. Rule-based:     keyword matching on IR fields → fast, zero-dependency
  2. Embedding-based: IR embedding vs dataset summary embeddings → medium, no LLM
  3. LLM-based:      structured prompt with IR + candidate summaries → most accurate
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# Domain Ontology (extensible — auto-discovers new domains from indexed data)
_BASE_DOMAINS = [
    "retail",          # e-commerce, sales, orders, products, customers
    "medical",         # healthcare, cancer, insurance, patients, treatment
    "housing",         # real estate, property, price, neighborhood, mortgage
    "transportation",  # taxi, ride, fare, trip, route, vehicle
    "education",       # MBA, admission, GPA, GMAT, university, school
    "iot",             # sensor, factory, machine, energy, manufacturing, IoT
    "wine",            # wine quality, alcohol, acidity, chemistry
    "animal",          # cat, dog, breed, species, wildlife
    "finance",         # insurance, investment, banking, stock, portfolio
    "sports",          # F1, football, racing, player, team, match, league
    "entertainment",   # superhero, card games, comics, movies, games
    "chemistry",       # toxicology, molecule, atom, bond, element
    "community",       # codebase, forum, posts, users, badges, reputation
    "general",         # generic or cross-domain
    "unknown",         # cannot determine
]


def get_allowed_domains() -> list[str]:
    """
    Build domain list dynamically: base ontology + domains discovered
    from indexed DatasetCards. This ensures new domains (genomics,
    astronomy
    """
    domains = list(_BASE_DOMAINS)
    try:
        from core.dataset_card import load_all_dataset_cards
        for card in load_all_dataset_cards():
            if card.domain and card.domain.lower() not in domains:
                domains.append(card.domain.lower())
    except Exception:
        pass
    return domains


ALLOWED_DOMAINS = get_allowed_domains()


@dataclass
class DomainRoutingResult:
   #Output of domain routing — a soft prior, not a hard decision.
    predicted_domain: str = "general"
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    alternative_domains: list[dict] = field(default_factory=list)  # [{domain, score}]
    ambiguous: bool = False
    routing_mode: str = "off"  # "rule" | "embedding" | "llm" | "off"

    def to_dict(self) -> dict:
        return {
            "predicted_domain": self.predicted_domain,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
            "alternative_domains": self.alternative_domains,
            "ambiguous": self.ambiguous,
            "routing_mode": self.routing_mode,
        }


# Mode 1: Rule-Based (fastest, zero dependency)
# Domain keyword signals — extracted from typical column names & query terms
_DOMAIN_SIGNALS: dict[str, list[str]] = {
    "retail": ["sales", "revenue", "order", "product", "customer", "payment",
               "cart", "sku", "discount", "invoice", "seller", "olist", "purchase"],
    "medical": ["cancer", "patient", "treatment", "diagnosis", "medical", "health",
                "disease", "mortality", "hospital", "drug", "smoking", "smoker",
                "bmi", "blood", "severity", "stage", "charges", "insurance"],
    "housing": ["house", "home", "price", "property", "rent", "mortgage", "bedroom",
                "neighborhood", "crim", "lstat", "medv", "rm", "chas", "boston",
                "residence", "sqft", "zoning"],
    "transportation": ["taxi", "trip", "fare", "ride", "cab", "passenger", "driver",
                        "route", "distance", "duration", "vehicle", "uber", "lyft"],
    "education": ["admission", "gmat", "gpa", "university", "mba", "student",
                  "academic", "school", "degree", "major", "enrollment", "graduate"],
    "iot": ["sensor", "machine", "temperature", "energy", "power", "factory",
            "iot", "manufacturing", "vibration", "pressure", "detector", "motor",
            "industrial", "equipment", "monitor", "consumption"],
    "wine": ["wine", "alcohol", "acidity", "sugar", "sulfur", "quality",
             "tannin", "vintage", "grape", "ph", "chloride", "density"],
    "animal": ["cat", "dog", "breed", "species", "animal", "feline", "pet",
               "weight", "lifespan", "fur", "paw"],
    "finance": ["stock", "portfolio", "investment", "bank", "interest", "rate",
                "bond", "equity", "dividend", "market", "fund", "return",
                "loan", "account", "transaction", "trans", "card", "disp",
                "district", "client", "currency", "consumption", "gasstation",
                "debit", "credit", "payment"],
    "sports": ["race", "racing", "driver", "circuit", "constructor", "lap", "laptime",
               "pit", "grid", "qualifying", "podium", "championship", "season",
               "standing", "fastest", "formula", "f1",
               "football", "soccer", "player", "team", "match", "league",
               "goal", "assist", "tackle", "penalty", "referee", "stadium",
               "rating", "potential", "fifa", "sprint", "overtake",
               "winner", "runner", "points", "pole", "driverId", "raceId"],
    "entertainment": ["superhero", "hero", "power", "superpower", "villain",
                      "comic", "publisher", "alignment", "attribute",
                      "card", "cards", "foil", "borderless", "rarity", "set",
                      "legality", "format", "ruling", "artist", "mana"],
    "chemistry": ["molecule", "atom", "bond", "element", "toxicology", "toxic",
                  "compound", "chemical", "reaction", "carcinogenic", "label",
                  "connected", "single", "double", "triple"],
    "community": ["post", "comment", "badge", "reputation", "user", "vote",
                  "tag", "answer", "question", "score", "view", "upvote",
                  "moderator", "wiki", "revision", "bounty", "codebase"],
}


def route_by_rules(intent_dict: dict) -> DomainRoutingResult:
    """
    Rule-based domain routing: count keyword hits in IR fields.

    Fast, deterministic, zero-dependency. Good baseline for ablation.
    """
    # Collect all text signals from IR
    all_hints = []
    all_hints.extend(intent_dict.get("metric_hints", []))
    all_hints.extend(intent_dict.get("dimension_hints", []))
    for fh in intent_dict.get("filter_hints", []):
        if isinstance(fh, dict):
            all_hints.append(fh.get("field_hint", ""))
        elif hasattr(fh, "field_hint"):
            all_hints.append(fh.field_hint)
    all_hints.extend(intent_dict.get("time_hints", []))
    # Also include raw query if available
    raw = intent_dict.get("raw_query", "")
    text = " ".join(all_hints + [raw]).lower()

    # Score each domain
    scores: dict[str, float] = {}
    evidence_map: dict[str, list[str]] = {}

    for domain, keywords in _DOMAIN_SIGNALS.items():
        hits = []
        for kw in keywords:
            if kw in text:
                hits.append(kw)
        score = len(hits) / max(len(keywords), 1)
        if hits:
            scores[domain] = score
            evidence_map[domain] = [f"keyword hit: {h}" for h in hits[:3]]

    if not scores:
        return DomainRoutingResult(
            predicted_domain="general", confidence=0.2,
            evidence=["no domain keywords detected in IR"],
            ambiguous=True, routing_mode="rule",
        )

    # Sort by score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_domain, best_score = ranked[0]

    # Check ambiguity: top-2 very close
    ambiguous = len(ranked) >= 2 and (ranked[0][1] - ranked[1][1]) < 0.05

    alternatives = [{"domain": d, "score": round(s, 3)} for d, s in ranked[1:3]]

    return DomainRoutingResult(
        predicted_domain=best_domain,
        confidence=round(min(best_score * 3, 0.95), 3),  # scale up, cap at 0.95
        evidence=evidence_map.get(best_domain, []),
        alternative_domains=alternatives,
        ambiguous=ambiguous,
        routing_mode="rule",
    )


# Mode 2: Embedding-Based (medium, no LLM call)
def route_by_embedding(intent_dict: dict) -> DomainRoutingResult:
    """
    Embedding-based domain routing: encode IR text, compare against
    dataset summary embeddings, infer domain from closest datasets.

    Uses pre-computed summary_embedding from DatasetCard (built during indexing).
    No LLM call needed — only local embedding similarity.
    """
    try:
        from core.dataset_card import load_all_dataset_cards
        from core.embedding_engine import get_encoder
        import numpy as np

        encoder = get_encoder()
        cards = load_all_dataset_cards()
        if not cards:
            return route_by_rules(intent_dict)  # fallback

        # Build query text from IR
        parts = []
        parts.extend(intent_dict.get("metric_hints", []))
        parts.extend(intent_dict.get("dimension_hints", []))
        raw = intent_dict.get("raw_query", "")
        if raw:
            parts.append(raw)
        query_text = " ".join(parts)

        if not query_text.strip():
            return route_by_rules(intent_dict)

        query_vec = encoder.encode([query_text])[0]

        # Score each dataset by embedding similarity
        dataset_scores = []
        for card in cards:
            if card.summary_embedding:
                card_vec = np.array(card.summary_embedding, dtype=np.float32)
                sim = float(encoder.similarity(query_vec, card_vec.reshape(1, -1))[0])
                dataset_scores.append((card.dataset_id, card.domain, sim))

        if not dataset_scores:
            return route_by_rules(intent_dict)

        dataset_scores.sort(key=lambda x: x[2], reverse=True)

        # Aggregate domain scores from top-5 datasets
        domain_scores: dict[str, float] = {}
        domain_evidence: dict[str, list[str]] = {}
        for ds_id, domain, sim in dataset_scores[:5]:
            dom = domain if domain else "general"
            domain_scores[dom] = domain_scores.get(dom, 0) + sim
            if dom not in domain_evidence:
                domain_evidence[dom] = []
            domain_evidence[dom].append(f"{ds_id} (sim={sim:.3f})")

        ranked = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        best_domain, best_score = ranked[0]

        # Normalize confidence
        total = sum(s for _, s in ranked)
        confidence = best_score / total if total > 0 else 0.3

        ambiguous = len(ranked) >= 2 and (ranked[0][1] - ranked[1][1]) / max(total, 1) < 0.1

        alternatives = [{"domain": d, "score": round(s / total, 3)} for d, s in ranked[1:3]]

        return DomainRoutingResult(
            predicted_domain=best_domain,
            confidence=round(confidence, 3),
            evidence=domain_evidence.get(best_domain, [])[:3],
            alternative_domains=alternatives,
            ambiguous=ambiguous,
            routing_mode="embedding",
        )

    except Exception as e:
        logger.warning(f"[DomainRouter] Embedding mode failed: {e}, falling back to rules")
        return route_by_rules(intent_dict)


# Mode 3: LLM-Based
_LLM_SYSTEM_PROMPT = """\
You are a domain routing module in a cross-domain data discovery system.

Your task is to infer the most likely domain of a user query from its structured intermediate representation (IR).

The inferred domain is used only as a soft routing signal for downstream schema grounding and column binding. It is NOT the final interpretation of the query.

Rules:
1. Use only the evidence explicitly present in the IR.
2. Do not invent fields, tables, business context, or user intent not supported by the IR.
3. Choose the most likely domain from the allowed domain list only.
4. If the IR is generic, underspecified, or domain-ambiguous, return "general" or "unknown".
5. Be conservative with confidence scores.
6. Domain inference is a routing prior, not a hard decision.
7. Prefer lexical and semantic evidence from metric_hints, dimension_hints, filters, agg_func_hint.
8. When multiple domains are plausible, mark ambiguous=true and provide alternatives.

Return JSON only:
{
  "predicted_domain": "<one allowed domain>",
  "confidence": <float 0..1>,
  "evidence": ["<short evidence 1>", "<short evidence 2>"],
  "alternative_domains": [{"domain": "<domain>", "score": <float>}],
  "ambiguous": <true|false>
}"""

_LLM_FEW_SHOT = """
Example 1:
IR: {"metric_hints":["charges"],"dimension_hints":["smoker"],"agg_func_hint":"avg"}
Output: {"predicted_domain":"medical","confidence":0.86,"evidence":["charges is common in healthcare/insurance","smoker is a health-related variable"],"alternative_domains":[{"domain":"finance","score":0.08}],"ambiguous":false}

Example 2:
IR: {"metric_hints":["house price"],"dimension_hints":["neighborhood"],"agg_func_hint":"avg"}
Output: {"predicted_domain":"housing","confidence":0.91,"evidence":["house price strongly suggests real estate","neighborhood is a housing grouping"],"alternative_domains":[{"domain":"finance","score":0.04}],"ambiguous":false}

Example 3:
IR: {"metric_hints":["energy consumption"],"dimension_hints":["machine type"],"agg_func_hint":"sum"}
Output: {"predicted_domain":"iot","confidence":0.89,"evidence":["energy consumption is industrial telemetry","machine type suggests equipment monitoring"],"alternative_domains":[{"domain":"general","score":0.08}],"ambiguous":false}

Example 4:
IR: {"metric_hints":["quality"],"dimension_hints":["alcohol"],"agg_func_hint":"avg"}
Output: {"predicted_domain":"wine","confidence":0.84,"evidence":["quality + alcohol combination is characteristic of wine data","avg by alcohol level suggests wine analysis"],"alternative_domains":[{"domain":"general","score":0.10}],"ambiguous":false}

Example 5:
IR: {"metric_hints":["score"],"dimension_hints":["category"],"agg_func_hint":null}
Output: {"predicted_domain":"general","confidence":0.41,"evidence":["score is domain-generic","category does not provide grounding"],"alternative_domains":[{"domain":"education","score":0.23},{"domain":"retail","score":0.21}],"ambiguous":true}
"""


def route_by_llm(intent_dict: dict, candidate_summaries: list[dict] = None) -> DomainRoutingResult:
    """
    LLM-based domain routing with structured prompt.

    Optionally includes candidate table summaries for stronger grounding.
    """
    try:
        import openai
        from config import LLM_MODEL, LLM_API_KEY, LLM_API_BASE

        api_key = LLM_API_KEY or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        api_base = LLM_API_BASE or os.getenv("LLM_API_BASE")

        if not api_key:
            return route_by_embedding(intent_dict)

        # Build user prompt
        ir_json = json.dumps({
            "intent_type": intent_dict.get("intent_type", "unknown"),
            "metric_hints": intent_dict.get("metric_hints", []),
            "dimension_hints": intent_dict.get("dimension_hints", []),
            "filter_hints": [f.get("field_hint", "") if isinstance(f, dict) else str(f)
                             for f in intent_dict.get("filter_hints", [])],
            "agg_func_hint": intent_dict.get("agg_func_hint"),
            "time_hints": intent_dict.get("time_hints", []),
        }, indent=2)

        user_parts = [
            f"Allowed domains:\n{json.dumps(ALLOWED_DOMAINS)}",
            f"\nIR:\n{ir_json}",
        ]

        # Add candidate table summaries if available (stronger grounding)
        if candidate_summaries:
            summaries_json = json.dumps(candidate_summaries[:5], indent=2)
            user_parts.append(f"\nCandidate table summaries:\n{summaries_json}")

        user_parts.append("\nInfer the most likely domain. Return JSON only.")
        user_prompt = "\n".join(user_parts)

        client_kwargs = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base

        client = openai.OpenAI(**client_kwargs, timeout=60)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.1,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT + _LLM_FEW_SHOT},
                {"role": "user", "content": user_prompt},
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

        domain = raw.get("predicted_domain", "general")
        if domain not in ALLOWED_DOMAINS:
            domain = "general"

        return DomainRoutingResult(
            predicted_domain=domain,
            confidence=min(raw.get("confidence", 0.5), 0.99),
            evidence=raw.get("evidence", [])[:3],
            alternative_domains=raw.get("alternative_domains", [])[:3],
            ambiguous=raw.get("ambiguous", False),
            routing_mode="llm",
        )

    except Exception as e:
        logger.warning(f"[DomainRouter] LLM mode failed: {e}, falling back to embedding")
        return route_by_embedding(intent_dict)


# Main entry
def route_domain(
    intent_dict: dict,
    mode: str = "auto",
    candidate_summaries: list[dict] = None,
) -> DomainRoutingResult:
    """
    Main domain routing entry
    Args:
        intent_dict: PartA output (QueryIntent as dict)
        mode: "rule" | "embedding" | "llm" | "auto" | "off"
        candidate_summaries: optional table summaries from PartB (for LLM mode)
    Returns:
        DomainRoutingResult with predicted_domain, confidence, evidence
    "auto" priority: embedding → rule (no LLM by default for speed)
    Use "llm" explicitly for highest accuracy.
    """
    if mode == "off":
        return DomainRoutingResult(routing_mode="off")

    if mode == "rule":
        return route_by_rules(intent_dict)

    if mode == "embedding":
        return route_by_embedding(intent_dict)

    if mode == "llm":
        return route_by_llm(intent_dict, candidate_summaries)

    # Auto: try embedding first (fast + good), fall back to rules
    result = route_by_embedding(intent_dict)
    if result.confidence < 0.3:
        # Embedding uncertain → try rules as backup
        rule_result = route_by_rules(intent_dict)
        if rule_result.confidence > result.confidence:
            return rule_result
    return result
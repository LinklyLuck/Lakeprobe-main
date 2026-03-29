"""
LakeProbe — Knowledge Base for Synonym Discovery
Function:
Replaces hardcoded ALIAS_LEXICON with dynamic synonym lookup.
Three-layer priority:
  1. ConceptNet API (online, richest, domain-aware)
  2. WordNet via NLTK (offline, linguistic synonyms)
  3. Extended JSON dictionary (bundled, covers common data domains)
  4. ALIAS_LEXICON fallback (hardcoded, last resort)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Cache to avoid repeated lookups
_synonym_cache: dict[str, list[str]] = {}

# Extended synonym dictionary path
_KB_DIR = Path(__file__).parent.parent / "data" / "knowledge_base"
_KB_DIR.mkdir(parents=True, exist_ok=True)
_EXTENDED_DICT_PATH = _KB_DIR / "synonyms.json"


# ConceptNet API (online, richest)
def _query_conceptnet(term: str, top_k: int = 10) -> list[str]:
    """
    Query ConceptNet 5.5 API for synonyms and related terms.
    Looks for /r/Synonym, /r/RelatedTo, /r/SimilarTo edges.
    Free API, no key needed, rate limit ~120 req/min.
    """
    try:
        import urllib.request
        import urllib.parse

        encoded = urllib.parse.quote(term.lower().replace("_", " "))
        url = f"http://api.conceptnet.io/c/en/{encoded}?limit=20"

        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())

        synonyms = set()
        target_relations = {"/r/Synonym", "/r/RelatedTo", "/r/SimilarTo", "/r/DerivedFrom"}

        for edge in data.get("edges", []):
            rel = edge.get("rel", {}).get("@id", "")
            if rel not in target_relations:
                continue

            # Extract the other end of the edge
            start = edge.get("start", {}).get("label", "").lower()
            end = edge.get("end", {}).get("label", "").lower()
            lang_start = edge.get("start", {}).get("language", "en")
            lang_end = edge.get("end", {}).get("language", "en")

            if lang_start == "en" and start != term.lower():
                synonyms.add(start)
            if lang_end == "en" and end != term.lower():
                synonyms.add(end)

        result = sorted(synonyms)[:top_k]
        if result:
            logger.info(f"[KB] ConceptNet: '{term}' → {result}")
        return result

    except Exception as e:
        logger.debug(f"[KB] ConceptNet unavailable for '{term}': {e}")
        return []


# WordNet via NLTK (offline, linguistic)
def _query_wordnet(term: str, top_k: int = 10) -> list[str]:
    """
    Query WordNet for synonyms
    """
    try:
        from nltk.corpus import wordnet as wn

        synonyms = set()
        for synset in wn.synsets(term.lower().replace("_", " ").replace(" ", "_")):
            for lemma in synset.lemmas():
                name = lemma.name().lower().replace("_", " ")
                if name != term.lower():
                    synonyms.add(name)

        result = sorted(synonyms)[:top_k]
        if result:
            logger.info(f"[KB] WordNet: '{term}' → {result}")
        return result

    except Exception as e:
        logger.debug(f"[KB] WordNet unavailable for '{term}': {e}")
        return []


# Extended JSON Dictionary
# Comprehensive domain-aware synonym dictionary
# Covers: business, finance, science, healthcare, education, tech, logistics, etc.
_EXTENDED_SYNONYMS: dict[str, list[str]] = {
    # ── Business & Sales ──
    "sales":        ["revenue", "income", "turnover", "earnings", "gross_sales", "proceeds", "receipts"],
    "revenue":      ["sales", "income", "turnover", "earnings", "gross_revenue", "top_line"],
    "profit":       ["margin", "net_income", "net_profit", "gain", "surplus", "earnings", "return"],
    "cost":         ["expense", "expenditure", "spending", "outlay", "price", "charge"],
    "price":        ["cost", "rate", "unit_price", "amount", "charge", "fee", "value"],
    "discount":     ["reduction", "rebate", "markdown", "deduction", "allowance"],
    "quantity":     ["qty", "count", "units", "volume", "amount", "number"],
    "order":        ["purchase", "transaction", "booking", "requisition"],
    "invoice":      ["bill", "receipt", "statement", "charge"],

    # ── People & Organizations ──
    "customer":     ["client", "user", "buyer", "account", "patron", "consumer", "purchaser"],
    "employee":     ["worker", "staff", "associate", "personnel", "team_member"],
    "supplier":     ["vendor", "provider", "manufacturer", "distributor"],
    "company":      ["organization", "firm", "enterprise", "business", "corporation"],
    "department":   ["division", "unit", "section", "group", "team"],
    "manager":      ["supervisor", "director", "head", "lead", "chief"],

    # ── Product & Inventory ──
    "product":      ["item", "sku", "goods", "merchandise", "article", "commodity"],
    "category":     ["type", "class", "group", "segment", "classification", "kind"],
    "brand":        ["make", "manufacturer", "label", "trademark"],
    "inventory":    ["stock", "supply", "warehouse", "holdings"],
    "description":  ["desc", "detail", "specification", "info", "summary"],

    # ── Geography ──
    "region":       ["area", "territory", "district", "zone", "geography", "locale"],
    "city":         ["town", "municipality", "urban_area", "metro"],
    "country":      ["nation", "state", "sovereign"],
    "address":      ["location", "place", "site", "venue"],
    "latitude":     ["lat", "y_coord"],
    "longitude":    ["lon", "lng", "x_coord"],

    # ── Time ──
    "date":         ["time", "period", "day", "month", "year", "quarter", "timestamp"],
    "year":         ["yr", "annual", "fiscal_year"],
    "month":        ["mo", "monthly", "period"],
    "quarter":      ["qtr", "q1", "q2", "q3", "q4", "fiscal_quarter"],
    "duration":     ["length", "span", "period", "interval", "time_span"],
    "age":          ["years_old", "lifetime", "maturity"],

    # ── Finance ──
    "balance":      ["amount", "total", "sum", "net"],
    "interest":     ["rate", "apr", "yield", "return"],
    "loan":         ["credit", "debt", "mortgage", "borrowing"],
    "payment":      ["installment", "remittance", "disbursement"],
    "tax":          ["levy", "duty", "tariff", "assessment"],
    "budget":       ["allocation", "funding", "appropriation"],
    "asset":        ["property", "holding", "resource", "investment"],

    # ── Science & Research ──
    "sample":       ["specimen", "observation", "instance", "case"],
    "experiment":   ["trial", "test", "study", "assay"],
    "measurement":  ["reading", "observation", "value", "result"],
    "concentration":["density", "level", "amount", "titer"],
    "temperature":  ["temp", "heat", "thermal"],
    "pressure":     ["force", "stress", "tension"],
    "weight":       ["mass", "heaviness", "load"],
    "height":       ["elevation", "altitude", "stature"],
    "length":       ["distance", "extent", "span", "size"],
    "width":        ["breadth", "thickness", "diameter"],
    "area":         ["surface", "region", "extent", "size"],
    "volume":       ["capacity", "amount", "quantity"],
    "density":      ["concentration", "compactness", "thickness"],
    "frequency":    ["rate", "occurrence", "count", "incidence"],
    "ratio":        ["proportion", "fraction", "percentage", "share"],
    "score":        ["rating", "grade", "mark", "value", "points"],
    "quality":      ["grade", "standard", "level", "rating", "class"],

    # ── Healthcare ──
    "patient":      ["subject", "case", "individual", "person"],
    "diagnosis":    ["condition", "disease", "disorder", "ailment"],
    "treatment":    ["therapy", "intervention", "procedure", "medication"],
    "symptom":      ["sign", "indicator", "manifestation"],
    "dosage":       ["dose", "amount", "quantity", "mg"],

    # ── Education ──
    "student":      ["learner", "pupil", "enrollee", "participant"],
    "grade":        ["score", "mark", "rating", "gpa"],
    "course":       ["class", "subject", "module", "program"],
    "enrollment":   ["registration", "admission", "signup"],

    # ── Technology ──
    "user":         ["customer", "member", "subscriber", "account"],
    "session":      ["visit", "interaction", "connection"],
    "click":        ["tap", "hit", "action", "interaction"],
    "conversion":   ["sale", "signup", "completion", "success"],
    "traffic":      ["visits", "hits", "page_views", "sessions"],
    "latency":      ["delay", "response_time", "lag", "wait_time"],
    "throughput":   ["bandwidth", "capacity", "rate", "speed"],
    "error":        ["fault", "failure", "bug", "defect", "issue"],
    "status":       ["state", "condition", "phase", "stage"],

    # ── Logistics ──
    "shipment":     ["delivery", "consignment", "dispatch", "package"],
    "warehouse":    ["depot", "storage", "facility", "distribution_center"],
    "tracking":     ["monitoring", "tracing", "surveillance"],
    "route":        ["path", "way", "itinerary", "journey"],

    # ── Wine / Food (domain-specific example) ──
    "alcohol":      ["abv", "ethanol", "alcohol_content", "spirit"],
    "acidity":      ["acid", "ph", "sourness", "tartness"],
    "sugar":        ["sweetness", "residual_sugar", "glucose", "sucrose"],
    "flavor":       ["taste", "aroma", "bouquet", "palate"],
    "color":        ["colour", "hue", "shade", "tint"],

    # ── General ──
    "name":         ["label", "title", "identifier", "tag"],
    "id":           ["identifier", "key", "code", "number", "index"],
    "type":         ["kind", "class", "category", "sort", "variety"],
    "level":        ["tier", "rank", "grade", "stage", "degree"],
    "rate":         ["ratio", "percentage", "proportion", "speed"],
    "total":        ["sum", "aggregate", "cumulative", "overall"],
    "average":      ["mean", "avg", "median", "typical"],
    "maximum":      ["max", "highest", "peak", "top", "upper"],
    "minimum":      ["min", "lowest", "bottom", "floor", "lower"],
    "count":        ["number", "quantity", "tally", "total", "frequency"],
    "percentage":   ["percent", "pct", "ratio", "share", "proportion"],
    "change":       ["delta", "difference", "variation", "shift", "movement"],
    "growth":       ["increase", "rise", "gain", "expansion", "appreciation"],
    "decline":      ["decrease", "drop", "fall", "reduction", "contraction"],
    "target":       ["goal", "objective", "label", "outcome", "dependent_variable"],
    "feature":      ["attribute", "variable", "predictor", "independent_variable", "column"],
    "result":       ["outcome", "output", "finding", "conclusion"],
}


def _query_extended_dict(term: str) -> list[str]:
    """Look up in extended bundled dictionary."""
    key = term.lower().strip().replace("_", " ").replace("-", " ")

    # Direct lookup
    if key in _EXTENDED_SYNONYMS:
        return _EXTENDED_SYNONYMS[key]

    # Try without spaces
    key_no_space = key.replace(" ", "")
    for k, v in _EXTENDED_SYNONYMS.items():
        if k.replace(" ", "") == key_no_space:
            return v

    # Reverse lookup: if term appears as a synonym value
    for canonical, syns in _EXTENDED_SYNONYMS.items():
        if key in [s.lower() for s in syns]:
            result = [canonical] + [s for s in syns if s.lower() != key]
            return result[:10]

    return []


# ALIAS_LEXICON Fallback
def _query_alias_lexicon(term: str) -> list[str]:
    """Last resort: original hardcoded ALIAS_LEXICON."""
    from config import ALIAS_LEXICON, ALIAS_REVERSE

    key = term.lower().strip()
    canonical = ALIAS_REVERSE.get(key)
    if canonical:
        return [canonical] + ALIAS_LEXICON.get(canonical, [])

    if key in ALIAS_LEXICON:
        return ALIAS_LEXICON[key]

    return []


# Main Entry Point
def get_synonyms(
    term: str,
    top_k: int = 10,
    use_conceptnet: bool = True,
    use_wordnet: bool = True,
) -> list[str]:
    """
    Get synonyms for a term using multi-layer knowledge base.
    Priority: cache → abbreviation → ConceptNet → WordNet → extended dict → ALIAS_LEXICON
    Returns deduplicated list of synonyms (not including the term itself).
    """
    key = term.lower().strip()

    # Check cache
    if key in _synonym_cache:
        return _synonym_cache[key]

    all_syns = set()

    # Layer 0: Abbreviation expansion (e.g., "qty" → "quantity", "cust" → "customer")
    abbr_expansions = expand_abbreviation(key)
    all_syns.update(abbr_expansions)
    # Also get abbreviation for full terms (e.g., "quantity" → "qty")
    abbr = get_abbreviation(key)
    if abbr:
        all_syns.add(abbr)

    # Layer 1: ConceptNet (online, richest)
    if use_conceptnet:
        cn_syns = _query_conceptnet(key)
        all_syns.update(cn_syns)

    # Layer 2: WordNet (offline, linguistic)
    if use_wordnet:
        wn_syns = _query_wordnet(key)
        all_syns.update(wn_syns)

    # Layer 3: Extended dictionary (bundled)
    ext_syns = _query_extended_dict(key)
    all_syns.update(ext_syns)

    # Layer 4: ALIAS_LEXICON (fallback)
    lex_syns = _query_alias_lexicon(key)
    all_syns.update(lex_syns)

    # Remove the term itself and empty strings
    all_syns.discard(key)
    all_syns.discard("")

    result = sorted(all_syns)[:top_k]
    _synonym_cache[key] = result
    return result


def get_canonical(term: str) -> str:
    """
    Get the canonical form of a term.
    """
    key = term.lower().strip()

    # Check extended dict (reverse lookup)
    for canonical, syns in _EXTENDED_SYNONYMS.items():
        all_forms = [canonical] + [s.lower() for s in syns]
        if key in all_forms:
            return canonical

    # Fallback to ALIAS_REVERSE
    from config import ALIAS_REVERSE
    return ALIAS_REVERSE.get(key, key)


def build_reverse_index() -> dict[str, str]:
    """
    Build a comprehensive reverse index: synonym → canonical.
    Merges ALIAS_LEXICON + extended dict.
    """
    reverse = {}

    # From extended dict
    for canonical, syns in _EXTENDED_SYNONYMS.items():
        reverse[canonical.lower()] = canonical
        for s in syns:
            reverse[s.lower()] = canonical

    # From ALIAS_LEXICON (lower priority, won't overwrite)
    from config import ALIAS_LEXICON
    for canonical, syns in ALIAS_LEXICON.items():
        if canonical.lower() not in reverse:
            reverse[canonical.lower()] = canonical
        for s in syns:
            if s.lower() not in reverse:
                reverse[s.lower()] = canonical

    return reverse


# Pre-built reverse index for fast lookup
EXTENDED_REVERSE: dict[str, str] = build_reverse_index()


# Abbreviation Expansion (for dirty data)
# Common data column abbreviations → full forms
_ABBREVIATIONS: dict[str, list[str]] = {
    # Quantities & measures
    "qty": ["quantity", "count", "units"],
    "amt": ["amount", "total", "value"],
    "val": ["value", "amount"],
    "num": ["number", "count"],
    "cnt": ["count", "number"],
    "tot": ["total", "sum"],
    "avg": ["average", "mean"],
    "pct": ["percent", "percentage", "ratio"],
    "rt": ["rate", "ratio"],
    "wt": ["weight"],
    "ht": ["height"],
    "len": ["length"],
    "sz": ["size"],
    "vol": ["volume"],
    "cap": ["capacity"],
    "freq": ["frequency"],
    "dur": ["duration"],
    "dist": ["distance"],
    "temp": ["temperature"],
    "prs": ["pressure"],
    "spd": ["speed"],

    # Finance
    "rev": ["revenue", "sales", "income"],
    "inc": ["income", "revenue"],
    "exp": ["expense", "expenditure", "cost"],
    "cst": ["cost", "expense"],
    "prc": ["price", "cost"],
    "bal": ["balance"],
    "pmt": ["payment"],
    "txn": ["transaction"],
    "inv": ["invoice", "inventory"],
    "mrg": ["margin", "profit margin"],
    "pnl": ["profit and loss", "pnl"],
    "roi": ["return on investment"],
    "cogs": ["cost of goods sold"],
    "ebitda": ["earnings before interest tax depreciation amortization"],

    # People & org
    "cust": ["customer", "client"],
    "usr": ["user", "customer"],
    "emp": ["employee", "staff"],
    "mgr": ["manager"],
    "dept": ["department", "division"],
    "org": ["organization", "company"],
    "acct": ["account"],
    "addr": ["address", "location"],
    "tel": ["telephone", "phone"],
    "ph": ["phone", "telephone"],

    # Product
    "prod": ["product", "item"],
    "sku": ["stock keeping unit", "product code"],
    "cat": ["category", "type", "class"],
    "grp": ["group", "category"],
    "seg": ["segment", "group"],
    "cls": ["class", "category", "type"],
    "typ": ["type", "category"],
    "brnd": ["brand"],
    "mdl": ["model"],
    "ver": ["version"],

    # Geography
    "rgn": ["region", "area"],
    "loc": ["location", "place"],
    "cty": ["city", "town"],
    "st": ["state", "street"],
    "ctry": ["country", "nation"],
    "zip": ["zipcode", "postal code"],
    "lat": ["latitude"],
    "lon": ["longitude"],
    "lng": ["longitude"],

    # Time
    "dt": ["date", "datetime"],
    "ts": ["timestamp", "datetime"],
    "yr": ["year"],
    "mo": ["month"],
    "dy": ["day"],
    "hr": ["hour"],
    "min": ["minute"],
    "sec": ["second"],
    "qtr": ["quarter"],
    "wk": ["week"],

    # Status & flags
    "sts": ["status"],
    "flg": ["flag", "indicator"],
    "ind": ["indicator", "index"],
    "lvl": ["level"],
    "pri": ["priority"],
    "src": ["source"],
    "tgt": ["target"],
    "ref": ["reference"],

    # Tech / data
    "id": ["identifier", "key"],
    "pk": ["primary key"],
    "fk": ["foreign key"],
    "idx": ["index"],
    "desc": ["description"],
    "lbl": ["label", "name"],
    "nm": ["name"],
    "cd": ["code"],
    "img": ["image"],
    "url": ["link", "address"],
    "msg": ["message"],
    "err": ["error"],
    "log": ["record", "entry"],
    "cfg": ["configuration", "config"],

    # Science
    "conc": ["concentration"],
    "ph": ["acidity", "pH level"],
    "abv": ["alcohol by volume", "alcohol content"],
    "rsd": ["residual"],
    "sgr": ["sugar"],
    "chl": ["chloride"],
    "sulf": ["sulfate", "sulphate"],
    "dens": ["density"],
    "acid": ["acidity"],
    "alc": ["alcohol"],
    "vol_acid": ["volatile acidity"],

    # Sports / Racing (BIRD: formula_1, european_football_2)
    "pos": ["position", "placement", "rank"],
    "pts": ["points", "score"],
    "lap": ["lap time", "circuit lap"],
    "gp": ["grand prix", "race"],
    "dnf": ["did not finish"],
    "ctor": ["constructor", "team"],
    "drv": ["driver", "racer"],
    "rnd": ["round"],
    "qual": ["qualifying", "qualification"],

    # Entertainment (BIRD: superhero, card_games)
    "pwr": ["power", "ability"],
    "attr": ["attribute", "property"],
    "hp": ["health points", "hit points"],
    "atk": ["attack"],
    "def": ["defense"],
    "dmg": ["damage"],

    # Community / Forum (BIRD: codebase_community)
    "rep": ["reputation", "score"],
    "upv": ["upvote", "like"],
    "ans": ["answer", "response"],
    "cmt": ["comment", "reply"],

    # Finance / Banking (BIRD: financial, debit_card_specializing)
    "xact": ["transaction"],
    "disp": ["disposition", "arrangement"],
    "stmt": ["statement"],
    "seg": ["segment", "category"],
    "curr": ["currency"],
    "cons": ["consumption", "usage"],

    # Chemistry (BIRD: toxicology)
    "mol": ["molecule", "molecular"],
    "elem": ["element"],
    "bnd": ["bond"],
    "cpd": ["compound"],

    # Education (BIRD: california_schools, student_club)
    "enrl": ["enrollment", "enrolled"],
    "sch": ["school"],
    "sat": ["SAT score", "scholastic aptitude"],
    "grd": ["grade", "level"],
    "evt": ["event"],
    "mbr": ["member"],
    "bgt": ["budget"],
}

# Build reverse: full_form → abbreviation
_ABBREV_REVERSE: dict[str, str] = {}
for abbr, fulls in _ABBREVIATIONS.items():
    for full in fulls:
        _ABBREV_REVERSE[full.lower()] = abbr


def expand_abbreviation(term: str) -> list[str]:
    """
    Expand a column name abbreviation to full forms
    """
    key = term.lower().strip()

    # Direct match
    if key in _ABBREVIATIONS:
        return _ABBREVIATIONS[key]

    # Try splitting compound abbreviations (e.g., "cust_seg" → ["cust", "seg"])
    tokens = key.replace("_", " ").replace("-", " ").split()
    if len(tokens) > 1:
        expanded = []
        for t in tokens:
            if t in _ABBREVIATIONS:
                expanded.extend(_ABBREVIATIONS[t][:1])  # take first expansion
            else:
                expanded.append(t)
        if expanded != tokens:  # something was expanded
            return expanded

    return []


def get_abbreviation(full_term: str) -> Optional[str]:
    """Get abbreviation for a full term. e.g., "quantity" → "qty" """
    return _ABBREV_REVERSE.get(full_term.lower().strip())
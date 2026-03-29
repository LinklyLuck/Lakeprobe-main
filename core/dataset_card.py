"""
LakeProbe — Part B: DatasetCard Builder + Column Index
Fuction:
Two-layer generation:
  First layer: Computable facts (from ProfileCard)
  Second layer: Summary generation (using a small LLM, optional)
Column Index: Create lexical, alias, role, and stats indexes for each column.
"""

from __future__ import annotations

import json
from pathlib import Path
from difflib import SequenceMatcher

from core.models import (
    ColumnIndexEntry,
    ColumnProfile,
    ColumnRole,
    DatasetCard,
    ProfileCard,
)
from config import (
    ALIAS_LEXICON,
    ALIAS_REVERSE,
    COLUMN_INDEX_DIR,
    DATASET_CARD_DIR,
)


# DatasetCard Builder
def build_dataset_card(profile: ProfileCard, save: bool = True) -> DatasetCard:
    """
    Build a DatasetCard from a ProfileCard.
    The first layer is based entirely on statistical facts from the profile.
    The second layer—purpose, domain, and summary—can be supplemented later using an LLM.
    """
    measure_cols = [c.name for c in profile.columns if c.inferred_role == ColumnRole.MEASURE]
    dimension_cols = [c.name for c in profile.columns if c.inferred_role == ColumnRole.DIMENSION]
    time_cols = [c.name for c in profile.columns if c.inferred_role == ColumnRole.TIME]
    all_names = [c.name for c in profile.columns]

    # Try to infer the domain from the column name
    domain = _infer_domain(all_names)
    entities = _infer_entities(dimension_cols)

    # Building the Foundation: Summary
    summary_parts = [
        f"Dataset '{profile.dataset_id}' with {profile.row_count} rows and {profile.col_count} columns.",
    ]
    if measure_cols:
        summary_parts.append(f"Measures: {', '.join(measure_cols)}.")
    if dimension_cols:
        summary_parts.append(f"Dimensions: {', '.join(dimension_cols)}.")
    if time_cols:
        summary_parts.append(f"Time columns: {', '.join(time_cols)}.")

    card = DatasetCard(
        dataset_id=profile.dataset_id,
        purpose=f"Data about {domain}" if domain else "",
        domain=domain,
        entities=entities,
        measure_columns=measure_cols,
        dimension_columns=dimension_cols,
        time_columns=time_cols,
        row_count=profile.row_count,
        column_names=all_names,
        summary=" ".join(summary_parts),
    )

    # Generate summary embedding for dataset-level semantic search
    # This is crucial for dirty data: even if table name is "tbl_0xf3a2",
    # the embedding captures column content (sample values, roles, types)
    try:
        from core.embedding_engine import get_encoder

        # Build rich embedding text: columns + aliases + sample values
        embed_parts = [f"dataset {profile.dataset_id}"]
        if domain:
            embed_parts.append(f"domain {domain}")
        for col in profile.columns:
            col_text = col.name.replace("_", " ")
            if col.sample_values:
                samples = ", ".join(str(v) for v in col.sample_values[:3])
                col_text += f" ({samples})"
            col_text += f" [{col.inferred_role.value}]"
            embed_parts.append(col_text)

        embed_text = " | ".join(embed_parts)
        encoder = get_encoder()
        vec = encoder.encode([embed_text])[0]
        card.summary_embedding = vec.tolist()
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"[DatasetCard] Embedding failed: {e}")

    if save:
        out_path = DATASET_CARD_DIR / f"{card.dataset_id}.json"
        out_path.write_text(card.model_dump_json(indent=2), encoding="utf-8")

    return card


def _infer_domain(col_names: list[str]) -> str:
    #Roughly infer the data domain based on keywords in the column names.
    name_set = {c.lower().replace("_", " ") for c in col_names}
    joined = " ".join(name_set)

    domain_keywords = {
        "sales":     ["sales", "revenue", "order", "transaction", "invoice"],
        "hr":        ["employee", "salary", "department", "hire", "job"],
        "finance":   ["profit", "loss", "balance", "expense", "cost", "budget"],
        "marketing": ["campaign", "click", "impression", "conversion", "channel"],
        "product":   ["product", "sku", "item", "inventory", "stock"],
        "customer":  ["customer", "client", "user", "subscriber", "member"],
        "logistics": ["shipment", "delivery", "warehouse", "tracking"],
    }

    scores = {}
    for domain, keywords in domain_keywords.items():
        score = sum(1 for kw in keywords if kw in joined)
        if score > 0:
            scores[domain] = score

    if scores:
        return max(scores, key=scores.get)
    return "general"


def _infer_entities(dimension_cols: list[str]) -> list[str]:
    #Deduce the entity name from the ‘dimension’ column name.
    entities = []
    for col in dimension_cols:
        clean = col.lower().replace("_id", "").replace("_name", "").replace("_", " ").strip()
        if clean and clean not in entities:
            entities.append(clean)
    return entities[:10]


# Column Index Builder (with Embedding)
def build_column_index(profile: ProfileCard, save: bool = True) -> list[ColumnIndexEntry]:
    """
    Create an index entry for each column in ProfileCard.
    Includes lexical key, alias list, role, dtype, statistical fingerprint, and dense embedding.
    Alias generation priority:
      1. Generated by the LLM (cached in data/alias_cache/)
      2. Fallback to the hard-coded ALIAS_LEXICON dictionary
    Embedding text construction strategy:
      “{col_name} | aliases: ... | role=... | type=... | values: ...”
    """
    from core.embedding_engine import (
        get_encoder, build_column_text, save_vectors,
    )
    from core.llm_alias import generate_aliases_for_dataset

    #Aliases in Bulk (LLM or Alternative Method)
    col_infos = [
        {
            "name": col.name,
            "dtype": col.dtype,
            "role": col.inferred_role.value,
            "samples": col.sample_values[:5],
        }
        for col in profile.columns
    ]
    all_aliases = generate_aliases_for_dataset(profile.dataset_id, col_infos)

    entries: list[ColumnIndexEntry] = []
    embed_texts: list[str] = []
    col_names: list[str] = []

    for col in profile.columns:
        lexical_key = col.name.lower().strip().replace("_", " ").replace("-", " ")

        # Retrieve aliases from LLM/cache/fallback
        aliases = all_aliases.get(col.name, [col.name.lower()])

        # Statistics
        stats_fp = {
            "missing_rate": col.missing_rate,
            "unique_rate": col.unique_rate,
            "n_unique": col.n_unique,
            "dtype": col.dtype,
        }
        if col.min_val is not None:
            stats_fp["min"] = col.min_val
        if col.max_val is not None:
            stats_fp["max"] = col.max_val

        # Construct an embedding from the input text
        embed_text = build_column_text(
            column_name=col.name,
            aliases=aliases,
            role=col.inferred_role.value,
            dtype=col.dtype,
            top_values=col.top_values,
            sample_values=col.sample_values,
        )
        embed_texts.append(embed_text)
        col_names.append(col.name)

        entry = ColumnIndexEntry(
            dataset_id=profile.dataset_id,
            column_name=col.name,
            lexical_key=lexical_key,
            aliases=aliases,
            inferred_role=col.inferred_role,
            dtype=col.dtype,
            stats_fingerprint=stats_fp,
        )
        entries.append(entry)

    # Batch-encode embeddings
    encoder = get_encoder()

    # TF-IDF must be trained first
    if hasattr(encoder, 'fit') and not getattr(encoder, '_is_fitted', True):
        encoder.fit(embed_texts)

    vectors = encoder.encode(embed_texts)

    # Write to each entry
    for i, entry in enumerate(entries):
        entry.embedding = vectors[i].tolist()

    # Persistence
    if save:
        # JSON index (include embedding list)
        out_path = COLUMN_INDEX_DIR / f"{profile.dataset_id}.json"
        data = [e.model_dump() for e in entries]
        out_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

        # NPZ vector index (high-performance dense search)
        save_vectors(profile.dataset_id, col_names, vectors, embed_texts)

    return entries


def _build_aliases(col_name: str) -> list[str]:
    #Generate a list of aliases for the specified column.
    base = col_name.lower().strip()
    aliases = set()

    # Original name
    aliases.add(base)

    # Remove the underline
    aliases.add(base.replace("_", " "))
    aliases.add(base.replace("_", ""))

    # If there is a corresponding canonical entry in the alias lexicon
    canonical = ALIAS_REVERSE.get(base)
    if canonical:
        aliases.add(canonical)
        # Include all synonyms for “canonical” as well
        for a in ALIAS_LEXICON.get(canonical, []):
            aliases.add(a.lower())

    # Parsing multi-word column names
    tokens = base.replace("_", " ").replace("-", " ").split()
    for t in tokens:
        canon = ALIAS_REVERSE.get(t)
        if canon:
            aliases.add(canon)

    return sorted(aliases - {""})

# Tool
def load_dataset_card(dataset_id: str) -> DatasetCard | None:
    path = DATASET_CARD_DIR / f"{dataset_id}.json"
    if not path.exists():
        return None
    return DatasetCard.model_validate_json(path.read_text(encoding="utf-8"))


def load_column_index(dataset_id: str) -> list[ColumnIndexEntry]:
    path = COLUMN_INDEX_DIR / f"{dataset_id}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ColumnIndexEntry(**d) for d in data]


def load_all_dataset_cards() -> list[DatasetCard]:
    cards = []
    for f in sorted(DATASET_CARD_DIR.glob("*.json")):
        try:
            cards.append(DatasetCard.model_validate_json(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return cards

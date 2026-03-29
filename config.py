"""
LakeProbe — Global Configuration
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
CSV_DIR = DATA_DIR / "csv"
PROFILE_DIR = DATA_DIR / "profile_cards"
DATASET_CARD_DIR = DATA_DIR / "dataset_cards"
LLM_TIMEOUT = 60
COLUMN_INDEX_DIR = DATA_DIR / "column_index"
VECTOR_INDEX_DIR = DATA_DIR / "vector_index"

# Ensure directories exist
for d in [CSV_DIR, PROFILE_DIR, DATASET_CARD_DIR, COLUMN_INDEX_DIR, VECTOR_INDEX_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# LLM configuration
LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-4o-mini"        # "GPT" | "gemini" | "deepseek"
LLM_TEMPERATURE = 0.1
LLM_API_BASE = "https://goapi.gptnb.ai/v1"
LLM_API_KEY = "sk-0mQ2NHGkJZTAmUsj2eB98477058f497fBd0aDcA74661823c"

# Retrieval configuration
DATASET_TOP_K = 3                # top-k results returned by dataset retrieval
COLUMN_TOP_K = 5                 # top-k results returned per hint in column retrieval
MIN_CANDIDATE_SCORE = 0.3        # candidates below this score are discarded directly

# ── Embedding configuration ──
EMBEDDING_MODE = "auto"          # "auto" | "openai" | "neural" | "tfidf"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # sentence-transformers model name
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model
OPENAI_EMBEDDING_DIM = 512      # reduce dimension to 512 (text-embedding-3-small supports 256/512/1536)
EMBEDDING_DIM_TFIDF = 128       # target dimension after TF-IDF SVD reduction
EMBEDDING_NGRAM_RANGE = (2, 4)  # character n-gram range
EMBEDDING_BATCH_SIZE = 50      # OpenAI API batch size

# ── Hybrid Retrieval 权重 ──
RETRIEVAL_WEIGHTS = {
    "sparse": 0.50,              # lexical + alias + role + dtype + stats
    "dense": 0.50,               # embedding cosine similarity
}

# Knowledge Base configuration (synonym discovery)
# Priority: WordNet (offline, free) → ConceptNet (online) → LLM → Extended Dict → ALIAS_LEXICON
KB_USE_WORDNET = True            # NLTK WordNet (offline, requires nltk.download('wordnet'))
KB_USE_CONCEPTNET = False        # ConceptNet API (online, slower, recommended off for demos)
KB_WORDNET_TOP_K = 8            # number of synonyms to take from WordNet per word
KB_CONCEPTNET_TOP_K = 6         # number of synonyms to take from ConceptNet per word

# Fusion configuration
FUSION_WEIGHTS = {
    "lexical": 0.25,
    "semantic": 0.30,            # upgraded: higher weight for embedding signal
    "role_match": 0.20,
    "dtype_compat": 0.15,
    "topk_evidence": 0.10,
}

# ── Three-zone threshold routing (from DataSearchTool / DBCopilot) ──
# Adapted from DBCopilot's Schema Routing threshold design:
#   reject zone   : score < REJECT_THRESHOLD  → auto-discard, not shown
#   uncertain zone: REJECT_THRESHOLD ≤ score < ACCEPT_THRESHOLD → needs user confirmation
#   accept zone   : score ≥ ACCEPT_THRESHOLD → auto-accept
BINDING_REJECT_THRESHOLD = 0.30
BINDING_ACCEPT_THRESHOLD = 0.70
BINDING_UNCERTAIN_CORE_LOW = 0.45   # lower bound of the core ambiguous zone (within uncertain zone)

# Execution engine
EXECUTOR_BACKEND = "duckdb"      # "duckdb" | "polars"
MAX_RESULT_ROWS = 500

# Domain Routing configuration
# Mode: "auto" (embedding→rule), "embedding", "rule", "llm", "off"
DOMAIN_ROUTING_MODE = "auto"
DOMAIN_BOOST_WEIGHT = 0.3        # added score weight for dataset ranking when domain matches

# ── Alias（fallback for offline scenarios when KB/LLM unavailable） ──
ALIAS_LEXICON: dict[str, list[str]] = {
    "revenue": ["sales", "income", "earnings", "turnover"],
    "sales": ["revenue", "income", "turnover"],
    "profit": ["earnings", "net income", "margin"],
    "price": ["cost", "amount", "value", "fare", "rate"],
    "quantity": ["qty", "count", "number", "volume"],
    "date": ["time", "day", "period", "timestamp"],
    "region": ["area", "zone", "territory", "location", "district"],
    "category": ["type", "class", "group", "segment"],
    "customer": ["client", "buyer", "user", "account"],
    "product": ["item", "goods", "merchandise", "sku"],
    "name": ["title", "label", "identifier"],
    "age": ["years old", "years"],
    "country": ["nation", "state", "territory"],
    "city": ["town", "municipality"],
    "gender": ["sex"],
    "score": ["rating", "grade", "mark", "points"],
    "weight": ["mass"],
    "height": ["altitude", "elevation"],
    "temperature": ["temp"],
    "energy": ["power", "consumption", "usage", "watt"],
    "speed": ["velocity", "rate"],
    "salary": ["wage", "pay", "compensation", "income"],
    "department": ["dept", "division", "unit"],
    "status": ["state", "condition"],
}

ALIAS_REVERSE: dict[str, str] = {}
for _canon, _aliases in ALIAS_LEXICON.items():
    for _a in _aliases:
        if _a not in ALIAS_REVERSE:
            ALIAS_REVERSE[_a] = _canon

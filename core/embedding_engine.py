"""
LakeProbe — Embedding Engine

Dual-Mode Dense Encoder:
  Mode A: sentence-transformers (full MiniLM-L6-v2) — Production-grade semantic retrieval
  Mode B: TF-IDF character n-gram projection — Lightweight fallback, zero dependencies
The external interfaces for both modes are identical:
  encode(texts) → np.ndarray[N, D]
  similarity(query_vec, corpus_mat) → np.ndarray[N]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from config import VECTOR_INDEX_DIR

logger = logging.getLogger(__name__)


# Abstract Encoder Interface
class BaseEncoder:
    #Encoder interface; subclasses must implement encode().
    dim: int = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def similarity(self, query_vec: np.ndarray, corpus_mat: np.ndarray) -> np.ndarray:
        """Cosine similarity: query_vec [D] × corpus_mat [N, D] → scores [N]"""
        if corpus_mat.shape[0] == 0:
            return np.array([])
        # Normalize
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        c_norms = corpus_mat / (np.linalg.norm(corpus_mat, axis=1, keepdims=True) + 1e-10)
        return c_norms @ q_norm



# Mode A: OpenAI Embeddings (fastest, API-based)
class OpenAIEmbeddingEncoder(BaseEncoder):
    #OpenAI text-embedding-3-small.
    def __init__(self, model: str = None, dim: int = None):
        import os
        from config import (
            LLM_API_KEY, LLM_API_BASE,
            OPENAI_EMBEDDING_MODEL, OPENAI_EMBEDDING_DIM, EMBEDDING_BATCH_SIZE,
        )

        self._model = model or OPENAI_EMBEDDING_MODEL
        self.dim = dim or OPENAI_EMBEDDING_DIM
        self._batch_size = EMBEDDING_BATCH_SIZE

        api_key = LLM_API_KEY or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        api_base = LLM_API_BASE or os.getenv("LLM_API_BASE")

        if not api_key:
            raise ValueError("No API key for OpenAI embeddings")

        import openai
        client_kwargs = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base
        self._client = openai.OpenAI(**client_kwargs, timeout=60)

        logger.info(f"[Embedding] OpenAI encoder: model={self._model}, dim={self.dim}")

    def encode(self, texts: list[str]) -> np.ndarray:
        #Batch encode via OpenAI API.
        all_embeddings = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            # Clean empty strings (API rejects them) and truncate long texts
            batch = [t[:500] if t.strip() else "empty" for t in batch]

            try:
                resp = self._client.embeddings.create(
                    model=self._model,
                    input=batch,
                    dimensions=self.dim,
                )
                batch_vecs = [item.embedding for item in resp.data]
                all_embeddings.extend(batch_vecs)
            except Exception as e:
                logger.warning(f"[Embedding] OpenAI batch {i} failed: {e}, padding zeros")
                all_embeddings.extend([[0.0] * self.dim] * len(batch))

        result = np.array(all_embeddings, dtype=np.float32)
        # L2 normalize
        norms = np.linalg.norm(result, axis=1, keepdims=True) + 1e-10
        return result / norms


# Mode B: Sentence-Transformers (local neural)
class SentenceTransformerEncoder(BaseEncoder):
  # Trained using sentence-transformers/all-MiniLM-L6-v2.
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"[Embedding] Loaded SentenceTransformer '{model_name}', dim={self.dim}")

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# Mode B: TF-IDF Character N-gram (lightweight)
class TfidfNgramEncoder(BaseEncoder):
  # Character n-gram encoding based on the sklearn TfidfVectorizer.
    def __init__(self, target_dim: int = 128, ngram_range: tuple = (2, 4)):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import Pipeline

        self.target_dim = target_dim
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            max_features=5000,
            sublinear_tf=True,
        )
        self._svd = TruncatedSVD(n_components=target_dim, random_state=42)
        self._pipeline: Optional[Pipeline] = None
        self._is_fitted = False
        self.dim = target_dim
        logger.info(f"[Embedding] TF-IDF char n-gram encoder, dim={target_dim}")

    def fit(self, corpus: list[str]) -> None:
        #Apply the TF-IDF + SVD pipeline to the corpus.
        if len(corpus) < 2:
            #padding
            corpus = corpus + ["placeholder column"] * max(2, self.target_dim - len(corpus))

        actual_dim = min(self.target_dim, len(corpus) - 1)
        if actual_dim != self._svd.n_components:
            from sklearn.decomposition import TruncatedSVD
            self._svd = TruncatedSVD(n_components=actual_dim, random_state=42)
            self.dim = actual_dim

        tfidf_mat = self._vectorizer.fit_transform(corpus)
        # Ensure that n_components ≤ n_features
        n_features = tfidf_mat.shape[1]
        actual_dim = min(actual_dim, n_features - 1) if n_features > 1 else 1
        if actual_dim != self._svd.n_components:
            from sklearn.decomposition import TruncatedSVD
            self._svd = TruncatedSVD(n_components=actual_dim, random_state=42)
            self.dim = actual_dim
        self._svd.fit(tfidf_mat)
        self._is_fitted = True
        logger.info(f"[Embedding] Fitted on {len(corpus)} texts, "
                     f"vocab={len(self._vectorizer.vocabulary_)}, "
                     f"explained_var={self._svd.explained_variance_ratio_.sum():.3f}")

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._is_fitted:
            self.fit(texts)
        tfidf_mat = self._vectorizer.transform(texts)
        dense = self._svd.transform(tfidf_mat)
        # L2 normalize
        norms = np.linalg.norm(dense, axis=1, keepdims=True) + 1e-10
        return dense / norms


# Column Embedding Text Builder
def build_column_text(
    column_name: str,
    aliases: list[str],
    role: str,
    dtype: str,
    top_values: list = None,
    sample_values: list = None,
) -> str:
    #    Construct the embedding for the column using the input text.
    #    Scheme: Column name + alias + role + type + sample value
    parts = [column_name.replace("_", " ")]

    if aliases:
        parts.append("aliases: " + ", ".join(aliases[:5]))

    parts.append(f"role={role}")
    parts.append(f"type={dtype}")

    if top_values:
        vals = [str(v) for v in top_values[:5]]
        parts.append("values: " + ", ".join(vals))
    elif sample_values:
        vals = [str(v) for v in sample_values[:5]]
        parts.append("samples: " + ", ".join(vals))

    return " | ".join(parts)


def build_hint_text(hint: str, hint_type: str = "", agg_func: str = "") -> str:
   # Construct the embedding input text for the query hint.
   # Strategy: hint keywords + role prompts + aggregation function

    parts = [hint]
    if hint_type:
        parts.append(f"role={hint_type}")
    if agg_func:
        parts.append(f"agg={agg_func}")
    return " | ".join(parts)


# Encoder Factory + Singleton
_encoder_instance: Optional[BaseEncoder] = None

def get_encoder(force_mode: str = None) -> BaseEncoder:
    """
    Get the global encoder instance (singleton pattern).

    Priority: OpenAI API (fastest) → sentence-transformers (local) → TF-IDF (fallback)
    force_mode: “openai” | ‘neural’ | “tfidf” | None (auto-detect)
    """
    global _encoder_instance
    if _encoder_instance is not None:
        return _encoder_instance

    if force_mode == "tfidf":
        _encoder_instance = TfidfNgramEncoder()
        return _encoder_instance

    if force_mode == "neural":
        _encoder_instance = SentenceTransformerEncoder()
        return _encoder_instance

    if force_mode == "openai":
        _encoder_instance = OpenAIEmbeddingEncoder()
        return _encoder_instance

    # Auto-detect: OpenAI → sentence-transformers → TF-IDF
    # 1. Try OpenAI (fastest)
    try:
        _encoder_instance = OpenAIEmbeddingEncoder()
        logger.info("[Embedding] Using OpenAI text-embedding-3-small (API)")
        return _encoder_instance
    except Exception as e:
        logger.info(f"[Embedding] OpenAI not available ({e}), trying local...")

    # 2. Try sentence-transformers (local)
    try:
        _encoder_instance = SentenceTransformerEncoder()
        logger.info("[Embedding] Using sentence-transformers (local)")
        return _encoder_instance
    except (ImportError, Exception) as e:
        logger.info(f"[Embedding] sentence-transformers not available ({e}), "
                     f"falling back to TF-IDF")

    # 3. TF-IDF fallback
    _encoder_instance = TfidfNgramEncoder()
    logger.info("[Embedding] Using TF-IDF n-gram encoder (fallback)")
    return _encoder_instance


def reset_encoder():
    #Testing
    global _encoder_instance
    _encoder_instance = None



# Vector Index Persistence
def save_vectors(dataset_id: str, column_names: list[str],
                 vectors: np.ndarray, texts: list[str]) -> Path:
    #Persist column vector indexes to disk.
    out_path = VECTOR_INDEX_DIR / f"{dataset_id}.npz"
    np.savez_compressed(
        str(out_path),
        vectors=vectors,
        column_names=np.array(column_names, dtype=object),
        texts=np.array(texts, dtype=object),
    )
    # Also update the ANN index if available
    try:
        ann = get_ann_index()
        ann.add_vectors(dataset_id, column_names, vectors)
    except Exception:
        pass
    return out_path


def load_vectors(dataset_id: str) -> Optional[dict]:
    #Load column vector indexes from disk.
    path = VECTOR_INDEX_DIR / f"{dataset_id}.npz"
    if not path.exists():
        return None
    data = np.load(str(path), allow_pickle=True)
    return {
        "vectors": data["vectors"],
        "column_names": data["column_names"].tolist(),
        "texts": data["texts"].tolist(),
    }


# ANN Index
class ANNIndex:
    """
    Approximate Nearest Neighbor index for scalable column vector search.
    Uses FAISS IVF index when available (thousands of datasets),
    falls back to exact brute-force for small collections.
    """

    def __init__(self, dim: int = 0):
        self.dim = dim
        self._faiss_index = None
        self._use_faiss = False
        self._vectors: list[np.ndarray] = []
        self._labels: list[tuple[str, str]] = []  # (dataset_id, column_name)
        self._built = False

        # Try to import FAISS
        try:
            import faiss
            self._faiss_module = faiss
            self._use_faiss = True
            logger.info("[ANN] FAISS available — using IVF index for scalable search")
        except ImportError:
            self._faiss_module = None
            self._use_faiss = False
            logger.info("[ANN] FAISS not available — using brute-force fallback")

    def add_vectors(self, dataset_id: str, column_names: list[str],
                    vectors: np.ndarray):
        """Add column vectors from a dataset to the index."""
        if vectors.ndim != 2 or vectors.shape[0] != len(column_names):
            return

        if self.dim == 0:
            self.dim = vectors.shape[1]

        for i, col_name in enumerate(column_names):
            self._vectors.append(vectors[i].astype(np.float32))
            self._labels.append((dataset_id, col_name))

        self._built = False  # needs rebuild

    def build(self):
        """Build/rebuild the ANN index from accumulated vectors."""
        if not self._vectors:
            return

        all_vecs = np.stack(self._vectors).astype(np.float32)
        n, d = all_vecs.shape

        if self._use_faiss and n >= 100:
            faiss = self._faiss_module
            # Use IVF index for large collections, flat for small
            if n >= 1000:
                nlist = min(int(np.sqrt(n)), 256)
                quantizer = faiss.IndexFlatIP(d)
                self._faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist,
                                                        faiss.METRIC_INNER_PRODUCT)
                # Normalize vectors for cosine similarity via inner product
                faiss.normalize_L2(all_vecs)
                self._faiss_index.train(all_vecs)
                self._faiss_index.add(all_vecs)
                self._faiss_index.nprobe = min(nlist // 4, 32)
            else:
                self._faiss_index = faiss.IndexFlatIP(d)
                faiss.normalize_L2(all_vecs)
                self._faiss_index.add(all_vecs)
        else:
            # Brute-force: just keep the stacked matrix
            norms = np.linalg.norm(all_vecs, axis=1, keepdims=True) + 1e-10
            self._brute_matrix = all_vecs / norms

        self._built = True
        logger.info(f"[ANN] Index built: {n} vectors, dim={d}, "
                     f"mode={'FAISS IVF' if self._use_faiss and n >= 1000 else 'FAISS Flat' if self._use_faiss else 'brute-force'}")

    def search(self, query_vec: np.ndarray, top_k: int = 20,
               dataset_filter: list[str] = None) -> list[tuple[str, str, float]]:
        """
        Search for nearest columns to query vector.
        Returns: [(dataset_id, column_name, similarity_score), ...]
        """
        if not self._built:
            self.build()

        if not self._vectors:
            return []

        q = query_vec.astype(np.float32).reshape(1, -1)

        if self._use_faiss and self._faiss_index is not None:
            faiss = self._faiss_module
            faiss.normalize_L2(q)
            k = min(top_k * 3, len(self._labels))  # over-fetch for filtering
            scores, indices = self._faiss_index.search(q, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                ds_id, col_name = self._labels[idx]
                if dataset_filter and ds_id not in dataset_filter:
                    continue
                results.append((ds_id, col_name, float(score)))
                if len(results) >= top_k:
                    break
            return results
        else:
            # Brute-force cosine similarity
            q_norm = q / (np.linalg.norm(q) + 1e-10)
            sims = (self._brute_matrix @ q_norm.T).flatten()

            indices = np.argsort(sims)[::-1]
            results = []
            for idx in indices:
                ds_id, col_name = self._labels[idx]
                if dataset_filter and ds_id not in dataset_filter:
                    continue
                results.append((ds_id, col_name, float(sims[idx])))
                if len(results) >= top_k:
                    break
            return results

    @property
    def size(self) -> int:
        return len(self._labels)


_ann_index: Optional[ANNIndex] = None


def get_ann_index() -> ANNIndex:
    """Get or create the global ANN index singleton"""
    global _ann_index
    if _ann_index is None:
        _ann_index = ANNIndex()
        # Load all existing vector files
        for npz_path in VECTOR_INDEX_DIR.glob("*.npz"):
            try:
                data = np.load(str(npz_path), allow_pickle=True)
                ds_id = npz_path.stem
                cols = data["column_names"].tolist()
                vecs = data["vectors"]
                _ann_index.add_vectors(ds_id, cols, vecs)
            except Exception:
                continue
        if _ann_index.size > 0:
            _ann_index.build()
            logger.info(f"[ANN] Loaded {_ann_index.size} vectors from disk")
    return _ann_index


def reset_ann_index():
    """Reset the ANN index (for reindexing)."""
    global _ann_index
    _ann_index = None
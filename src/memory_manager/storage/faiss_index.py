from __future__ import annotations

"""
FAISS index wrapper for fast candidate generation.

Implements a persistent cosine-similarity index over `nodes.embedding_blob`:
- Uses `faiss.IndexFlatIP` with pre-normalized vectors (dot product == cosine).
- Uses `faiss.IndexIDMap2` so FAISS IDs map back to `nodes.id`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.memory_manager.storage.memory_store import MemoryStore


@dataclass(frozen=True)
class FaissIndexConfig:
    node_types: Sequence[str] = ("concept",)
    index_path: str | None = None


def _require_faiss():
    try:
        import faiss  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "FAISS is not installed. Install with `pip install faiss-cpu`."
        ) from exc


def _load_embedding_blob(blob: bytes) -> np.ndarray:
    return np.asarray(np.frombuffer(blob, dtype=np.float32), dtype=np.float32)


def _normalize_rows(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms <= 0.0] = 1.0
    return vecs / norms


def build_faiss_flatip_index_from_db(
    *,
    store: MemoryStore,
    cfg: FaissIndexConfig | None = None,
) -> object:
    """
    Build a persistent FAISS IndexFlatIP over normalized embeddings.

    Returns:
        faiss index instance (typed as object to avoid hard import in typing).
    """
    _require_faiss()
    import faiss  # type: ignore

    cfg = cfg or FaissIndexConfig()
    allowed_types = {t.strip().lower() for t in cfg.node_types if t and t.strip()}

    rows = [
        r
        for r in store.fetch_all_nodes()
        if str(r["type"] or "").strip().lower() in allowed_types and r["embedding_blob"] is not None
    ]
    if not rows:
        raise RuntimeError("No nodes with embeddings found for FAISS index build.")

    ids = np.asarray([int(r["id"]) for r in rows], dtype=np.int64)
    vecs = np.asarray([_load_embedding_blob(r["embedding_blob"]) for r in rows], dtype=np.float32)
    if vecs.ndim != 2 or vecs.shape[0] != len(ids):
        raise RuntimeError("Invalid embedding vectors shape for FAISS build.")

    vecs = _normalize_rows(vecs)
    dim = int(vecs.shape[1])

    # Exact cosine via normalized vectors + inner product.
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
    index.add_with_ids(vecs, ids)

    index_path = _default_index_path(store) if cfg.index_path is None else str(cfg.index_path)
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    return index


def load_faiss_index(
    *,
    store: MemoryStore,
    cfg: FaissIndexConfig | None = None,
) -> object:
    _require_faiss()
    import faiss  # type: ignore

    cfg = cfg or FaissIndexConfig()
    index_path = _default_index_path(store) if cfg.index_path is None else str(cfg.index_path)
    return faiss.read_index(index_path)


def _default_index_path(store: MemoryStore) -> str:
    # Store index next to the DB file.
    assert store.cfg is not None
    return str(Path(store.cfg.db_path).parent / "faiss.index")


def faiss_search_topk(
    *,
    index: object,
    query_vec: np.ndarray,
    top_k: int,
) -> list[int]:
    """
    Search FAISS with a normalized query vector.
    Returns:
        list of node ids (int), length <= top_k.
    """
    import faiss  # type: ignore  # noqa: F401

    top_k = max(1, int(top_k))
    q = np.asarray(query_vec, dtype=np.float32)
    q = q.reshape(1, -1)

    # Normalize to ensure dot product == cosine similarity.
    q = _normalize_rows(q)

    D, I = index.search(q, int(top_k))
    out: list[int] = []
    for nid in I[0].tolist():
        if int(nid) < 0:
            continue
        out.append(int(nid))
    return out


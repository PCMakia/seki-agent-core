from __future__ import annotations

"""
Graph-aware memory retrieval over the SQLite concept graph.

Implements the requested graph-memory semantics:
- HEAD detection (embedding similarity against base nodes)
- bounded traversal HEAD -> cluster -> concept (using undirected adjacency)
- graph-distance + activation-weight reranking within a small candidate set
- episode evidence assembly via `episode_entities`
"""

from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from src.memory_manager.storage.memory_store import MemoryStore


@dataclass(frozen=True)
class GraphRetrieverConfig:
    embedding_model_name: str = "all-MiniLM-L6-v2"

    traversal_depth: int = 2
    max_candidate_concepts: int = 40

    rerank_top_m: int = 20
    max_graph_distance_depth: int = 2

    w_emb: float = 1.0
    w_act: float = 0.2
    w_dist: float = 0.5

    semantic_nodes_limit: int = 8
    episode_evidence_limit: int = 6

    # Approximate ROM/RAM tiering via in-process hot-head caching.
    head_cache_size: int = 32

    edge_relation_types: tuple[str, ...] = ("cluster_of", "in_cluster", "anchored_to", "co_occurs")


@dataclass(frozen=True)
class GraphRetrieveResult:
    block: str
    snippets: list[str]
    head_name: str | None
    selected_concept_names: list[str]


def _normalize(vec: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(vec))
    if denom <= 0.0:
        return vec
    return vec / denom


def _load_embedding_blob(blob: bytes) -> np.ndarray:
    vec = np.frombuffer(blob, dtype=np.float32)
    # Copy to detach from underlying buffer if needed.
    return np.asarray(vec, dtype=np.float32)


class GraphRetriever:
    def __init__(self, *, store: MemoryStore, cfg: GraphRetrieverConfig | None = None):
        self.store = store
        self.cfg = cfg or GraphRetrieverConfig()

        self._model = None
        self._head_ids: list[int] = []
        self._head_names: list[str] = []
        self._head_vecs: np.ndarray | None = None

        self._adj: dict[int, set[int]] = {}
        self._head_bfs_cache: OrderedDict[int, frozenset[int]] = OrderedDict()
        self._head_cache_size = max(1, int(self.cfg.head_cache_size))
        self._build_adjacency()
        self._load_head_embeddings()

    def _ensure_model(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(self.cfg.embedding_model_name)

    def _build_adjacency(self) -> None:
        allowed = {t.strip().lower() for t in self.cfg.edge_relation_types if t and t.strip()}
        edges = self.store.fetch_edges()

        adj: dict[int, set[int]] = {}
        for e in edges:
            rt = str(e["relation_type"] or "").strip().lower()
            if rt not in allowed:
                continue
            s = int(e["src_id"])
            d = int(e["dst_id"])
            adj.setdefault(s, set()).add(d)
            adj.setdefault(d, set()).add(s)
        self._adj = adj

    def _load_head_embeddings(self) -> None:
        head_rows = [
            r
            for r in self.store.fetch_all_nodes()
            if str(r["type"] or "").strip().lower() == "base" and r["embedding_blob"] is not None
        ]
        head_rows = sorted(head_rows, key=lambda r: (str(r["name"] or "").strip().lower(), int(r["id"])))
        if not head_rows:
            self._head_ids = []
            self._head_names = []
            self._head_vecs = None
            return

        ids = [int(r["id"]) for r in head_rows]
        names = [str(r["name"] or "").strip().lower() for r in head_rows]
        vecs = np.asarray([_load_embedding_blob(r["embedding_blob"]) for r in head_rows], dtype=np.float32)
        vecs = np.asarray([_normalize(v) for v in vecs], dtype=np.float32)

        self._head_ids = ids
        self._head_names = names
        self._head_vecs = vecs

    def detect_head(self, query: str) -> tuple[int | None, str | None]:
        """
        Detect the most relevant HEAD/base node for the query using cosine
        similarity in embedding space.
        """
        query = (query or "").strip()
        if not query:
            return None, None

        if self._head_vecs is None or not self._head_ids:
            # Fail-open: fall back to any base node (no embedding similarity).
            base_rows = [
                r for r in self.store.fetch_all_nodes() if str(r["type"] or "").strip().lower() == "base"
            ]
            if not base_rows:
                return None, None
            base_rows = sorted(base_rows, key=lambda r: int(r["id"]))
            return int(base_rows[0]["id"]), str(base_rows[0]["name"] or "").strip().lower()

        self._ensure_model()
        assert self._model is not None
        q_vec = np.asarray(self._model.encode([query])[0], dtype=np.float32)
        q_vec = _normalize(q_vec)

        sims = (self._head_vecs @ q_vec).reshape(-1)  # cosine with normalized vectors
        best_idx = int(np.argmax(sims))
        return self._head_ids[best_idx], self._head_names[best_idx]

    def _bfs_nodes(self, start_id: int) -> set[int]:
        sid = int(start_id)
        cached = self._head_bfs_cache.get(sid)
        if cached is not None:
            self._head_bfs_cache.move_to_end(sid)
            return set(cached)

        visited: set[int] = {sid}
        q: deque[tuple[int, int]] = deque([(sid, 0)])
        max_depth = max(0, int(self.cfg.traversal_depth))

        while q:
            nid, depth = q.popleft()
            if depth >= max_depth:
                continue
            for nei in self._adj.get(nid, set()):
                if nei in visited:
                    continue
                visited.add(nei)
                q.append((nei, depth + 1))
        visited.discard(int(start_id))
        frozen = frozenset(visited)

        self._head_bfs_cache[sid] = frozen
        self._head_bfs_cache.move_to_end(sid)
        while len(self._head_bfs_cache) > self._head_cache_size:
            self._head_bfs_cache.popitem(last=False)

        return set(frozen)

    def _graph_distance_to_targets(
        self,
        *,
        source_id: int,
        target_ids: set[int],
        max_depth: int,
    ) -> int:
        if int(source_id) not in self._adj:
            return max_depth + 1
        if not target_ids:
            return max_depth + 1

        start_id = int(source_id)
        visited: set[int] = {start_id}
        q: deque[tuple[int, int]] = deque([(start_id, 0)])

        while q:
            nid, depth = q.popleft()
            if depth >= max_depth:
                continue
            for nei in self._adj.get(nid, set()):
                if nei in visited:
                    continue
                if int(nei) in target_ids and int(nei) != start_id:
                    return depth + 1
                visited.add(int(nei))
                q.append((int(nei), depth + 1))
        return max_depth + 1

    def retrieve_context(
        self,
        *,
        session_id: str,
        user_text: str,
        episode_evidence_limit: int | None = None,
        semantic_nodes_limit: int | None = None,
    ) -> GraphRetrieveResult:
        session_id = (session_id or "default").strip() or "default"
        user_text = (user_text or "").strip()

        head_id, head_name = self.detect_head(user_text)
        if head_id is None:
            # No heads exist; return empty memory block.
            return GraphRetrieveResult(block="", snippets=[], head_name=None, selected_concept_names=[])

        visited = self._bfs_nodes(int(head_id))
        if not visited:
            return GraphRetrieveResult(block="", snippets=[], head_name=head_name, selected_concept_names=[])

        # Candidate concept nodes live inside the traversal neighborhood.
        nodes = self.store.fetch_nodes_by_ids(visited)
        concept_rows = [r for r in nodes if str(r["type"] or "").strip().lower() == "concept"]
        concept_rows.sort(key=lambda r: int(r["id"]))

        # Load query embedding once for reranking.
        self._ensure_model()
        assert self._model is not None
        q_vec = np.asarray(self._model.encode([user_text])[0], dtype=np.float32)
        q_vec = _normalize(q_vec)

        scored: list[tuple[float, int]] = []
        for r in concept_rows:
            emb_blob = r["embedding_blob"]
            if emb_blob is None:
                continue
            c_vec = _load_embedding_blob(emb_blob)
            c_vec = _normalize(c_vec)
            sim = float(np.dot(q_vec, c_vec))
            scored.append((sim, int(r["id"])))

        scored.sort(key=lambda x: (-x[0], x[1]))
        if not scored:
            return GraphRetrieveResult(block="", snippets=[], head_name=head_name, selected_concept_names=[])

        top_candidates = scored[: int(self.cfg.max_candidate_concepts)]
        candidate_ids = {nid for _, nid in top_candidates}

        top_m = top_candidates[: int(self.cfg.rerank_top_m)]
        top_m_ids = {nid for _, nid in top_m}

        # Build maps for candidate info.
        info_rows = self.store.fetch_nodes_by_ids(top_m_ids)
        info_by_id = {int(r["id"]): r for r in info_rows}

        # Rerank with graph distance.
        results: list[tuple[float, int]] = []
        for sim, nid in top_m:
            r = info_by_id.get(int(nid))
            if r is None:
                continue
            act = float(r["activation_weight"] or 0.0)
            dist = self._graph_distance_to_targets(
                source_id=int(nid),
                target_ids=top_m_ids,
                max_depth=int(self.cfg.max_graph_distance_depth),
            )
            score = float(self.cfg.w_emb) * sim + float(self.cfg.w_act) * act - float(self.cfg.w_dist) * float(dist)
            results.append((score, int(nid)))

        results.sort(key=lambda x: (-x[0], x[1]))
        sem_limit = int(semantic_nodes_limit) if semantic_nodes_limit is not None else int(self.cfg.semantic_nodes_limit)
        chosen_ids = [nid for _, nid in results[:sem_limit]]

        if not chosen_ids:
            return GraphRetrieveResult(block="", snippets=[], head_name=head_name, selected_concept_names=[])

        chosen_rows = self.store.fetch_nodes_by_ids(chosen_ids)
        chosen_rows.sort(key=lambda r: chosen_ids.index(int(r["id"])))

        selected_concept_names: list[str] = []
        semantic_lines: list[str] = []
        for r in chosen_rows:
            name = str(r["name"] or "").strip()
            summary = str(r["summary"] or "").strip()
            if not name:
                continue
            selected_concept_names.append(name)
            if summary:
                semantic_lines.append(f"- {name}: {summary}")
            else:
                semantic_lines.append(f"- {name}")

        evidence_snippets = self.store.fetch_recent_episode_snippets_for_node_ids(
            session_id=session_id,
            node_ids=chosen_ids,
            limit=int(episode_evidence_limit) if episode_evidence_limit is not None else int(self.cfg.episode_evidence_limit),
        )

        parts: list[str] = []
        if semantic_lines:
            parts.append("Semantic nodes:")
            parts.extend(semantic_lines)
        if evidence_snippets:
            parts.append("Recent episodes:")
            parts.extend([f"- {s}" for s in evidence_snippets])

        block = "\n".join(parts).strip()
        return GraphRetrieveResult(
            block=block,
            snippets=evidence_snippets,
            head_name=head_name,
            selected_concept_names=selected_concept_names,
        )

    def get_candidate_concept_ids(self, *, user_text: str, top_k: int) -> list[int]:
        """
        Return up to `top_k` concept node ids ranked by embedding similarity
        within the bounded HEAD traversal neighborhood.

        This is useful for the reasoning chain so it can start from graph
        traversal candidates instead of purely keyword-based SQL seeding.
        """
        top_k = max(1, int(top_k))
        user_text = (user_text or "").strip()
        if not user_text:
            return []

        head_id, _ = self.detect_head(user_text)
        if head_id is None:
            return []

        visited = self._bfs_nodes(int(head_id))
        if not visited:
            return []

        nodes = self.store.fetch_nodes_by_ids(visited)
        concept_rows = [r for r in nodes if str(r["type"] or "").strip().lower() == "concept"]
        if not concept_rows:
            return []

        # Build query embedding once.
        self._ensure_model()
        assert self._model is not None
        q_vec = np.asarray(self._model.encode([user_text])[0], dtype=np.float32)
        q_vec = _normalize(q_vec)

        scored: list[tuple[float, int]] = []
        for r in concept_rows:
            emb_blob = r["embedding_blob"]
            if emb_blob is None:
                continue
            c_vec = _normalize(_load_embedding_blob(emb_blob))
            sim = float(np.dot(q_vec, c_vec))
            scored.append((sim, int(r["id"])))

        scored.sort(key=lambda x: (-x[0], x[1]))
        return [nid for _, nid in scored[:top_k]]


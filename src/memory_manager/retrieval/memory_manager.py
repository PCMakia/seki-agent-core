""" MemoryManager class
"""

from __future__ import annotations


from dataclasses import dataclass
from datetime import datetime, timezone
import os
from typing import Any, Iterable, Optional

from collections import deque

from src.memory_manager.storage.memory_store import MemoryStore, tokenize
from src.memory_manager.retrieval.graph_memory_retriever import GraphRetriever, GraphRetrieverConfig


@dataclass(frozen=True)
class MemoryRetrieveResult:
    block: str
    snippets: list[str]


@dataclass(frozen=True)
class MemoryMetricsSample:
    ts: str
    session_id: str
    user_text: str
    clsm_block_preview: str
    reply_preview: str
    user_tokens: int
    clsm_tokens: int
    reply_tokens: int
    clsm_to_user_ratio: float
    clsm_to_reply_ratio: float
    overlap_tokens: int
    overlap_ratio_vs_reply: float


class MemoryManager:
    """
    Thin service layer over MemoryStore.

    Retrieves a compact "memory block"
    (episode snippets + any matching node summaries) and records each turn
    as an episode after the assistant replies.

    It also maintains a small in-memory buffer of recent "context efficiency"
    metrics so the API can expose lightweight debug information.
    """

    def __init__(self, store: Optional[MemoryStore] = None):
        self.store = store or MemoryStore()
        # Graph-memory retriever (fail-open: retrieval may fall back to the
        # previous keyword-based strategy if anything goes wrong).
        #
        # Toggle:
        # - set `MEMORY_USE_GRAPH_RETRIEVER=0` to force legacy keyword-based memory.
        self.use_graph_retriever = str(os.getenv("MEMORY_USE_GRAPH_RETRIEVER", "1")).strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        self.graph_retriever = (
            GraphRetriever(store=self.store, cfg=GraphRetrieverConfig())
            if self.use_graph_retriever
            else None
        )
        # Most recent context-efficiency samples (fail-open if anything goes wrong).
        self._metrics_buffer: deque[MemoryMetricsSample] = deque(maxlen=64)

    def retrieve_context(self, *, session_id: str, user_text: str, limit: int = 6) -> MemoryRetrieveResult:
        session_id = (session_id or "default").strip() or "default"
        user_text = (user_text or "").strip()

        if self.graph_retriever is not None:
            try:
                result = self.graph_retriever.retrieve_context(
                    session_id=session_id,
                    user_text=user_text,
                    episode_evidence_limit=limit,
                )
                if result.block:
                    return MemoryRetrieveResult(block=result.block, snippets=result.snippets)
            except Exception:
                # Fall back to the previous keyword-based retrieval to keep the
                # app resilient against any graph-memory regressions.
                pass

        # Legacy keyword-based retrieval fallback.
        tokens = tokenize(user_text)
        nodes = self.store.fetch_nodes_for_keyword_seeding(tokens, limit=12)
        node_names = [str(r["name"]) for r in nodes if (r.get("name") if hasattr(r, "get") else r["name"])]
        if not node_names:
            node_names = tokens[:6]

        snippets = self.store.fetch_recent_episode_snippets_for_node_names(
            session_id=session_id, node_names=node_names, limit=limit
        )

        node_summary_lines: list[str] = []
        for r in nodes[:8]:
            name = (r["name"] or "").strip()
            summary = (r["summary"] or "").strip()
            if not name:
                continue
            if summary:
                node_summary_lines.append(f"- {name}: {summary}")
            else:
                node_summary_lines.append(f"- {name}")

        parts: list[str] = []
        if node_summary_lines:
            parts.append("Semantic nodes:")
            parts.extend(node_summary_lines)
        if snippets:
            parts.append("Recent episodes:")
            parts.extend([f"- {s}" for s in snippets])

        block = "\n".join(parts).strip()
        return MemoryRetrieveResult(block=block, snippets=snippets)

    def record_interaction(
        self,
        *,
        session_id: str,
        user_text: str,
        assistant_text: str,
        usage: Any | None = None,
        importance: float = 0.0,
        topic: str | None = None,
    ) -> int:
        session_id = (session_id or "default").strip() or "default"
        return self.store.save_episode(
            session_id=session_id,
            user_text=(user_text or "").strip(),
            assistant_text=(assistant_text or "").strip(),
            usage=usage,
            importance=float(importance),
            topic=topic,
        )

    # --- Metrics & debug helpers -------------------------------------------------

    def record_usage_metrics(
        self,
        *,
        session_id: str,
        user_text: str,
        clsm_block: str,
        reply_text: str,
    ) -> None:
        """Record a single context-efficiency sample for debug/observability.

        All computations are intentionally lightweight and best-effort: any
        failure is swallowed to avoid impacting latency or correctness.
        """
        try:
            session_id = (session_id or "default").strip() or "default"
            user_text = (user_text or "").strip()
            clsm_block = (clsm_block or "").strip()
            reply_text = (reply_text or "").strip()

            user_toks = tokenize(user_text)
            clsm_toks = tokenize(clsm_block)
            reply_toks = tokenize(reply_text)

            user_count = len(user_toks)
            clsm_count = len(clsm_toks)
            reply_count = len(reply_toks)

            clsm_to_user = float(clsm_count) / float(user_count or 1)
            clsm_to_reply = float(clsm_count) / float(reply_count or 1)

            clsm_set = set(clsm_toks)
            overlap = [t for t in reply_toks if t in clsm_set]
            overlap_count = len(overlap)
            overlap_ratio_vs_reply = float(overlap_count) / float(reply_count or 1)

            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            sample = MemoryMetricsSample(
                ts=ts,
                session_id=session_id,
                user_text=user_text[:280],
                clsm_block_preview=clsm_block[:400],
                reply_preview=reply_text[:400],
                user_tokens=user_count,
                clsm_tokens=clsm_count,
                reply_tokens=reply_count,
                clsm_to_user_ratio=clsm_to_user,
                clsm_to_reply_ratio=clsm_to_reply,
                overlap_tokens=overlap_count,
                overlap_ratio_vs_reply=overlap_ratio_vs_reply,
            )
            self._metrics_buffer.append(sample)
        except Exception:
            # Metrics must never break chat; fail silently.
            return

    def get_recent_metrics(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return a JSON-serializable list of the most recent metric samples."""
        try:
            if limit <= 0:
                return []
            # Newest samples are at the right end of the deque.
            sliced = list(self._metrics_buffer)[-int(limit) :]
            out: list[dict[str, Any]] = []
            for s in reversed(sliced):
                out.append(
                    {
                        "ts": s.ts,
                        "session_id": s.session_id,
                        "user_tokens": s.user_tokens,
                        "clsm_tokens": s.clsm_tokens,
                        "reply_tokens": s.reply_tokens,
                        "clsm_to_user_ratio": s.clsm_to_user_ratio,
                        "clsm_to_reply_ratio": s.clsm_to_reply_ratio,
                        "overlap_tokens": s.overlap_tokens,
                        "overlap_ratio_vs_reply": s.overlap_ratio_vs_reply,
                        "user_text_preview": s.user_text,
                        "clsm_block_preview": s.clsm_block_preview,
                        "reply_preview": s.reply_preview,
                    }
                )
            return out
        except Exception:
            return []


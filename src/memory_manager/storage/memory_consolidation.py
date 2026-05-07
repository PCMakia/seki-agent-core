from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterable, Optional

from src.memory_manager.storage.memory_store import MemoryStore, tokenize, _utc_now_iso


@dataclass(frozen=True)
class ConsolidationConfig:
    """Runtime configuration for the consolidation worker."""

    interval_seconds: float = 60.0
    batch_size: int = 20
    max_tokens_per_episode: int = 20
    cooccurrence_relation_type: str = "co_occurs"
    anchoring_relation_type: str = "anchored_to"
    decay_factor: float = 0.98
    min_weight: float = 0.05


class ConsolidationWorker:
    """Periodic worker that turns episodes into knowledge-graph updates.

    Behaviour:
    - Fetch a batch of unconsolidated episodes.
    - Extract simple keyword-style tokens from user/assistant text.
    - Upsert concept nodes for those tokens.
    - Add co-occurrence edges between tokens that appeared in the same episode
      (both directions to approximate an undirected relation).
    - Mark processed episodes as consolidated.
    - Apply simple decay-based forgetting to existing node activations and
      edge weights.
    """

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        cfg: Optional[ConsolidationConfig] = None,
    ):
        self.store = store or MemoryStore()
        self.cfg = cfg or ConsolidationConfig()
        self._task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None

    async def start(self) -> None:
        """Start the background consolidation loop."""
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run_loop(), name="memory_consolidation_worker")

    async def stop(self) -> None:
        """Request the background loop to stop and wait for it."""
        if self._task is None or self._task.done() or self._stop_event is None:
            return
        self._stop_event.set()
        try:
            await self._task
        except Exception:
            # Fail-closed: worker shutdown must not crash the server.
            pass

    async def _run_loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:
                print(f"[memory] consolidation run failed: {exc}")

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.cfg.interval_seconds
                )
            except asyncio.TimeoutError:
                # Normal case: timeout means it's time for the next run.
                continue

    def run_once(self) -> None:
        """Execute a single consolidation cycle."""
        episodes = self.store.fetch_unconsolidated_episodes(
            limit=self.cfg.batch_size
        )
        now_ts = _utc_now_iso()

        if episodes:
            processed_ids: list[int] = []
            for ep in episodes:
                ep_id = int(ep["id"])
                user_text = (ep["user_text"] or "").strip()
                assistant_text = (ep["assistant_text"] or "").strip()
                text = f"{user_text}\n{assistant_text}"

                tokens = tokenize(text)
                unique_tokens = self._select_tokens(tokens)
                if not unique_tokens:
                    processed_ids.append(ep_id)
                    continue

                node_ids = self._upsert_nodes(unique_tokens)

                # Link this episode to its selected concept nodes so retrieval can
                # assemble evidence deterministically from `episode_entities`.
                # Note: `episode_entities` references the `entities` table, so we
                # mirror concept nodes into `entities(type='concept', name=token)`.
                for tok, _node_id in zip(unique_tokens, node_ids):
                    ent_id = self.store.upsert_entity(name=tok, type_="concept")
                    self.store.link_episode_entity(episode_id=ep_id, entity_id=ent_id)

                self._upsert_cooccurrence_edges(node_ids, now_ts)
                self._apply_base_anchoring(unique_tokens, node_ids, now_ts)
                processed_ids.append(ep_id)

            self.store.mark_episodes_consolidated(processed_ids, ts=now_ts)

        # Apply forgetting even if there was nothing new to consolidate.
        self.store.decay_graph(
            decay=self.cfg.decay_factor, min_weight=self.cfg.min_weight
        )

    def _select_tokens(self, tokens: Iterable[str]) -> list[str]:
        """Lightweight keyword/entity selection from raw tokens.

        Heuristic:
        - Lowercased tokens are already provided by tokenize().
        - Drop very short tokens (length < 3).
        - Keep only the first occurrence of each token.
        - Cap to a per-episode maximum to bound graph growth.
        """
        seen: set[str] = set()
        out: list[str] = []
        for t in tokens:
            t = t.strip()
            if len(t) < 3:
                continue
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
            if len(out) >= self.cfg.max_tokens_per_episode:
                break
        return out

    def _upsert_nodes(self, tokens: Iterable[str]) -> list[int]:
        node_ids: list[int] = []
        for tok in tokens:
            node_id = self.store.upsert_node(name=tok, type_="concept")
            node_ids.append(node_id)
        return node_ids

    def _upsert_cooccurrence_edges(self, node_ids: list[int], ts: str) -> None:
        n = len(node_ids)
        if n < 2:
            return
        relation = self.cfg.cooccurrence_relation_type
        for i in range(n):
            src = node_ids[i]
            for j in range(i + 1, n):
                dst = node_ids[j]
                # Store co-occurrence as bidirectional edges to approximate
                # an undirected relation using the existing directed schema.
                self.store.upsert_edge(
                    src_id=src,
                    dst_id=dst,
                    relation_type=relation,
                    weight_delta=1.0,
                    ts=ts,
                )
                self.store.upsert_edge(
                    src_id=dst,
                    dst_id=src,
                    relation_type=relation,
                    weight_delta=1.0,
                    ts=ts,
                )

    def _apply_base_anchoring(
        self, unique_tokens: list[str], node_ids: list[int], ts: str
    ) -> None:
        """Link non-base episode concepts to co-occurring base concepts (same token set)."""
        base_ids = self.store.fetch_base_node_ids_for_token_names(unique_tokens)
        if not base_ids:
            return
        types_by_id = self.store.fetch_node_types_by_ids(node_ids)
        relation = (self.cfg.anchoring_relation_type or "anchored_to").strip().lower()
        for nid in node_ids:
            ntype = str(types_by_id.get(nid, "concept")).strip().lower()
            if ntype == "base":
                continue
            for bid in base_ids:
                if nid == bid:
                    continue
                self.store.upsert_edge(
                    src_id=nid,
                    dst_id=bid,
                    relation_type=relation,
                    weight_delta=1.0,
                    ts=ts,
                )
                # Store the reverse direction too, so traversal can treat this
                # relation as effectively undirected without extra SQL logic.
                self.store.upsert_edge(
                    src_id=bid,
                    dst_id=nid,
                    relation_type=relation,
                    weight_delta=1.0,
                    ts=ts,
                )


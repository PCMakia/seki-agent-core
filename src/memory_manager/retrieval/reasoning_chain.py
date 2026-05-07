from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Optional

from src.memory_manager.storage.memory_store import MemoryStore, tokenize
from src.memory_manager.retrieval.graph_memory_retriever import GraphRetriever, GraphRetrieverConfig


def format_reasoning_block_text(
    result: Optional["ReasoningChainResult"] = None,
    *,
    error: Optional[str] = None,
    max_step_lines: int = 32,
    max_relation_lines: int = 12,
    max_evidence_lines: int = 6,
) -> str:
    """Compact, deterministic text for LLM prompt injection (no LLM calls).

    Presents the thought as an explicit head-to-tail linked sequence (system-built);
    the model is instructed to verbalize this chain only, not to extend it.
    """
    if error:
        return f"(reasoning unavailable: {error})"
    if result is None or not result.steps:
        return "(no reasoning steps)"

    lines: list[str] = []
    cap = max(1, int(max_step_lines))
    steps_slice = result.steps[:cap]
    # Singly-linked view: ordered nodes the runtime already ranked for this turn.
    chain_labels: list[str] = []
    for s in steps_slice:
        label = (s.name or "").strip() or "(unnamed)"
        chain_labels.append(f"[{s.step_type}] {label}")
    chain_line = " -> ".join(chain_labels)
    if len(result.steps) > cap:
        chain_line += f" -> ... ({len(result.steps) - cap} more)"
    lines.append(f"LINKED_CHAIN head-to-tail (system order; translate, do not reorder for new logic): {chain_line}")

    if result.tokens:
        tok_cap = min(24, len(result.tokens))
        lines.append(f"QUERY_TOKENS (retrieval hooks): {' '.join(result.tokens[:tok_cap])}")

    for s in steps_slice:
        lines.append(
            f"- [{s.step_type}] {s.name} | type={s.concept_type} | score={s.score:.2f} | {s.summary[:120]}"
        )
    if len(result.steps) > cap:
        lines.append(f"... ({len(result.steps) - cap} more steps omitted)")

    if result.relations:
        lines.append("GRAPH_LINKS (edges among concepts; optional nuance when wording connectivity):")
        rcap = max(1, int(max_relation_lines))
        for r in result.relations[:rcap]:
            lines.append(
                f"  {r.src_name} -[{r.relation_type}]-> {r.dst_name} (w={r.weight:.2f})"
            )
        if len(result.relations) > rcap:
            lines.append(f"  ... ({len(result.relations) - rcap} more omitted)")

    if result.evidence:
        lines.append("Evidence:")
        ecap = max(1, int(max_evidence_lines))
        for e in result.evidence[:ecap]:
            lines.append(f"  - {e[:200]}")
        if len(result.evidence) > ecap:
            lines.append(f"  ... ({len(result.evidence) - ecap} more omitted)")

    return "\n".join(lines)


@dataclass(frozen=True)
class ConceptStep:
    step_type: str
    node_id: int | None
    name: str
    concept_type: str
    summary: str
    matched_text: str
    position: int
    score: float


@dataclass(frozen=True)
class EdgeView:
    src_id: int
    src_name: str
    dst_id: int
    dst_name: str
    relation_type: str
    weight: float


@dataclass(frozen=True)
class ReasoningChainResult:
    session_id: str
    input_text: str
    tokens: list[str]
    steps: list[ConceptStep]
    relations: list[EdgeView]
    evidence: list[str]
    cache_hits: int = 0

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "input_text": self.input_text,
            "tokens": list(self.tokens),
            "steps": [asdict(s) for s in self.steps],
            "relations": [asdict(r) for r in self.relations],
            "evidence": list(self.evidence),
            "cache_hits": int(self.cache_hits),
        }


@dataclass(frozen=True)
class ReasoningConfig:
    max_candidate_nodes: int = 30
    max_steps: int = 10
    max_edges: int = 24
    max_evidence: int = 6
    topic_head_size: int = 3
    topic_cache_size: int = 32
    phrase_scan_limit: int = 400


class TopicHeadCache:
    """In-memory cache for recent chain heads (process lifetime only)."""

    def __init__(self, *, max_sessions: int = 128, head_history: int = 32):
        self._max_sessions = max(1, int(max_sessions))
        self._head_history = max(1, int(head_history))
        self._store: dict[str, deque[tuple[int, str]]] = defaultdict(
            lambda: deque(maxlen=self._head_history)
        )

    def put(self, *, session_id: str, heads: list[tuple[int, str]]) -> None:
        sid = (session_id or "default").strip() or "default"
        dq = self._store[sid]
        for node_id, name in heads:
            dq.append((int(node_id), (name or "").strip()))

        # Keep map bounded by evicting oldest session key if needed.
        if len(self._store) > self._max_sessions:
            oldest = next(iter(self._store.keys()))
            if oldest != sid:
                self._store.pop(oldest, None)

    def get_recent(self, *, session_id: str, limit: int = 6) -> list[tuple[int, str]]:
        sid = (session_id or "default").strip() or "default"
        dq = self._store.get(sid)
        if not dq:
            return []
        n = max(1, int(limit))
        return list(dq)[-n:]


class ReasoningChainEngine:
    """Deterministic reasoning chain over SQLite-backed concept graph.

    No LLM calls are made in this module.
    """

    def __init__(
        self,
        *,
        store: Optional[MemoryStore] = None,
        cache: Optional[TopicHeadCache] = None,
        cfg: Optional[ReasoningConfig] = None,
    ):
        self.store = store or MemoryStore()
        self.cache = cache or TopicHeadCache()
        self.cfg = cfg or ReasoningConfig()
        self.graph_retriever = GraphRetriever(
            store=self.store,
            cfg=GraphRetrieverConfig(),
        )

    def build_chain(self, *, session_id: str, text: str) -> ReasoningChainResult:
        sid = (session_id or "default").strip() or "default"
        user_text = (text or "").strip()
        raw_tokens = tokenize(user_text)
        tokens = list(dict.fromkeys(raw_tokens))

        command_steps = self._command_steps(user_text)
        phrase_terms = self._extract_phrase_terms(user_text)

        retrieval_terms = list(dict.fromkeys([*tokens, *phrase_terms]))
        nodes = []
        try:
            candidate_ids = self.graph_retriever.get_candidate_concept_ids(
                user_text=user_text,
                top_k=self.cfg.max_candidate_nodes,
            )
            if candidate_ids:
                nodes = self.store.fetch_nodes_by_ids(candidate_ids)
        except Exception:
            nodes = []

        # Fail-open fallback: preserve previous keyword-based SQL seeding if
        # graph traversal doesn't yield any candidates.
        if not nodes:
            nodes = self.store.fetch_nodes_for_keyword_seeding(
                retrieval_terms,
                limit=self.cfg.max_candidate_nodes,
            )
            if not nodes and tokens:
                nodes = self.store.fetch_nodes_for_keyword_seeding(
                    tokens[:8],
                    limit=self.cfg.max_candidate_nodes,
                )

        cached_ids = {nid for nid, _ in self.cache.get_recent(session_id=sid, limit=10)}
        lower_text = user_text.lower()

        scored: list[tuple[float, int, object]] = []
        for r in nodes:
            name = str(r["name"] or "").strip()
            if not name:
                continue
            name_l = name.lower()
            summary = str(r["summary"] or "").strip()
            pos = lower_text.find(name_l)
            summary_overlap = sum(1 for t in tokens if t in summary.lower())

            score = 0.0
            if pos >= 0:
                score += 3.0
            if name_l in tokens:
                score += 2.0
            score += min(2.0, 0.25 * float(summary_overlap))
            if int(r["id"]) in cached_ids:
                score += 1.5

            effective_pos = pos if pos >= 0 else 10_000_000
            scored.append((score, effective_pos, r))

        scored.sort(
            key=lambda x: (-x[0], x[1], str(x[2]["name"] or "").lower()),
        )

        concept_steps: list[ConceptStep] = []
        seen_node_ids: set[int] = set()
        for score, pos, r in scored:
            if len(concept_steps) >= self.cfg.max_steps:
                break
            node_id = int(r["id"])
            if node_id in seen_node_ids:
                continue
            seen_node_ids.add(node_id)
            name = str(r["name"] or "").strip()
            concept_type = str(r["type"] or "concept").strip().lower()
            summary = str(r["summary"] or "").strip()
            matched_text = name if lower_text.find(name.lower()) >= 0 else ""
            concept_steps.append(
                ConceptStep(
                    step_type="concept",
                    node_id=node_id,
                    name=name,
                    concept_type=concept_type,
                    summary=summary,
                    matched_text=matched_text,
                    position=pos if pos < 10_000_000 else -1,
                    score=float(score),
                )
            )

        steps = [*command_steps, *concept_steps]

        node_ids = [s.node_id for s in concept_steps if s.node_id is not None]
        edges_rows = self.store.fetch_edges_for_node_ids(
            node_ids,
            limit=self.cfg.max_edges,
        )
        chosen_node_ids = {int(nid) for nid in node_ids}
        edge_node_ids: set[int] = set()
        for e in edges_rows:
            edge_node_ids.add(int(e["src_id"]))
            edge_node_ids.add(int(e["dst_id"]))
        types_by_id = self.store.fetch_node_types_by_ids(edge_node_ids)
        relations: list[EdgeView] = []
        for e in edges_rows:
            src_id = int(e["src_id"])
            dst_id = int(e["dst_id"])
            src_type = str(types_by_id.get(src_id, "")).strip().lower()
            dst_type = str(types_by_id.get(dst_id, "")).strip().lower()
            keep = src_id in chosen_node_ids and dst_id in chosen_node_ids
            if not keep:
                keep = (src_id in chosen_node_ids and dst_type == "base") or (
                    dst_id in chosen_node_ids and src_type == "base"
                )
            if not keep:
                continue
            relations.append(
                EdgeView(
                    src_id=src_id,
                    src_name=str(e["src_name"] or "").strip(),
                    dst_id=dst_id,
                    dst_name=str(e["dst_name"] or "").strip(),
                    relation_type=str(e["relation_type"] or "").strip().lower(),
                    weight=float(e["weight"] or 0.0),
                )
            )

        evidence_names = [s.name for s in concept_steps[: self.cfg.max_evidence] if s.name]
        evidence = self.store.fetch_recent_episode_snippets_for_node_names(
            session_id=sid,
            node_names=evidence_names,
            limit=self.cfg.max_evidence,
        )

        heads = [(s.node_id, s.name) for s in concept_steps[: self.cfg.topic_head_size] if s.node_id]
        if heads:
            self.cache.put(session_id=sid, heads=heads)

        cache_hits = sum(
            1 for s in concept_steps if s.node_id is not None and int(s.node_id) in cached_ids
        )

        return ReasoningChainResult(
            session_id=sid,
            input_text=user_text,
            tokens=tokens,
            steps=steps,
            relations=relations,
            evidence=evidence,
            cache_hits=cache_hits,
        )

    def _extract_phrase_terms(self, text: str) -> list[str]:
        """Longest-phrase hints from known node names with spaces."""
        lower_text = (text or "").lower()
        if not lower_text:
            return []
        names = self.store.fetch_node_names_for_phrase_scan(limit=self.cfg.phrase_scan_limit)
        phrases: list[str] = []
        for name in names:
            n = (name or "").strip().lower()
            if not n or " " not in n:
                continue
            if n in lower_text:
                phrases.append(n)
        return phrases

    def _command_steps(self, text: str) -> list[ConceptStep]:
        raw = (text or "").strip()
        if not raw.startswith("/"):
            return []
        cmd = raw.split(maxsplit=1)[0].lower()
        cmd_summaries = {
            "/work": "Work mode command: optimize for actionable completion.",
            "/discuss": "Discuss mode command: optimize for trade-offs and understanding.",
            "/banter": "Banter mode command: friendly style while staying useful.",
        }
        summary = cmd_summaries.get(cmd)
        if not summary:
            return []
        return [
            ConceptStep(
                step_type="command",
                node_id=None,
                name=cmd,
                concept_type="command",
                summary=summary,
                matched_text=cmd,
                position=0,
                score=10.0,
            )
        ]


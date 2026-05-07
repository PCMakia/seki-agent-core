"""Just-in-time web lookup for user-mentioned concepts missing from the graph.

Runs **before** the secretary prompt is built (adds latency when enabled).
When ``MEMORY_JIT_WEB=1``, extracts likely entity phrases from the user message,
skips terms already backed by a node with a real summary (or already in the
reasoning chain with content), searches the web, writes/updates ``nodes.summary``,
links the new/updated concept to the detected topic **head** via ``co_occurs``,
and returns text to inject into the prompt.

Env:
  MEMORY_JIT_WEB=1|0              — master switch (default off)
  MEMORY_JIT_WEB_MAX_LOOKUPS=n    — max web round-trips per message (default 2)
  MEMORY_JIT_WEB_SCRAPE_TOP_N=n   — pages per lookup (default 3)
  MEMORY_JIT_WEB_STRICT=1|0       — require ``?`` + what/who/which + narrow patterns (default 1)
"""

from __future__ import annotations

import os
import re
import logging

from src.memory_manager.retrieval.graph_memory_retriever import GraphRetriever
from src.LLM_handler.intent_policy import is_narrow_definition_question
from src.memory_manager.storage.memory_store import MemoryStore
from src.memory_manager.retrieval.reasoning_chain import ReasoningChainResult

_LOG = logging.getLogger("personal_assistant.jit_web")

_STOPWORDS = frozenset(
    """
    the a an and or but if then else for to of in on at by is are was were be been being
    it its this that these those i you we they he she my your our their what which who how
    when where why can could should would will do does did has have had not no yes so as
    from with into about over under again further once here there all both each few more
    most other some such than too very just also only own same than too very
    """.split()
)


def jit_web_enabled() -> bool:
    return os.getenv("MEMORY_JIT_WEB", "").strip().lower() in ("1", "true", "yes", "on")


def _max_lookups() -> int:
    try:
        return max(0, min(6, int(os.getenv("MEMORY_JIT_WEB_MAX_LOOKUPS", "2"))))
    except ValueError:
        return 2


def _scrape_top_n() -> int:
    try:
        return max(1, min(8, int(os.getenv("MEMORY_JIT_WEB_SCRAPE_TOP_N", "3"))))
    except ValueError:
        return 3


def jit_web_strict_mode() -> bool:
    """When True, only definitional questions (``?`` + what/who/which) trigger JIT."""
    return os.getenv("MEMORY_JIT_WEB_STRICT", "1").strip().lower() not in ("0", "false", "no", "off")


def _is_placeholder_summary(summary: str) -> bool:
    s = (summary or "").strip().lower()
    if not s:
        return True
    if s.startswith("seeded from") or s.startswith("(seeded"):
        return True
    return False


# Strict: only phrases captured from what/who/which … ? (no quotes / TitleCase / raw tokens).
_STRICT_JIT_PATTERNS: tuple[str, ...] = (
    r"(?i)\bwhat\s+is\s+([^?]+)\?",
    r"(?i)\bwhat\s+are\s+([^?]+)\?",
    r"(?i)\bwhat\s+was\s+([^?]+)\?",
    r"(?i)\bwhat\s+were\s+([^?]+)\?",
    r"(?i)\bwho\s+is\s+([^?]+)\?",
    r"(?i)\bwho\s+are\s+([^?]+)\?",
    r"(?i)\bwho\s+was\s+([^?]+)\?",
    r"(?i)\bwhich\s+([^?]+)\?",
)


def _collect_phrase_candidates(user_text: str, *, strict: bool, pool_limit: int = 24) -> list[str]:
    """Extract entity strings to look up on the web.

    ``strict=True``: only ``what|who|which … ?`` regex captures (fewer false hits).
    ``strict=False``: legacy heuristics (quotes, Title Case, long tokens, tell/explain).
    """
    text = (user_text or "").strip()
    out: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        s = raw.strip().strip(".,;:!?\"'").strip()
        if len(s) < 2 or len(s) > 120:
            return
        k = s.lower()
        if k in seen:
            return
        seen.add(k)
        out.append(s)

    for pat in _STRICT_JIT_PATTERNS:
        for m in re.finditer(pat, text):
            add(m.group(1))

    if strict:
        return out[:pool_limit]

    loose_patterns = [
        r"(?i)\btell\s+me\s+about\s+([^?.!\n]+)",
        r"(?i)\bexplain\s+([^?.!\n]+)",
        r"(?i)\bdefine\s+([^?.!\n]+)",
    ]
    for pat in loose_patterns:
        for m in re.finditer(pat, text):
            add(m.group(1))

    for m in re.finditer(r'"([^"]{3,100})"|\'([^\']{3,100})\'', text):
        add(m.group(1) or m.group(2))

    for m in re.finditer(r"\b(?:[A-Z][a-z]+\s+){1,5}[A-Z][a-z]+\b", text):
        add(m.group(0))

    from src.memory_manager.storage.memory_store import tokenize

    for t in tokenize(text):
        if len(t) < 4 or t in _STOPWORDS:
            continue
        add(t)

    return out[:pool_limit]


def _known_from_reasoning(result: ReasoningChainResult | None) -> set[str]:
    known: set[str] = set()
    if not result:
        return known
    for s in result.steps:
        if (s.step_type or "").lower() != "concept":
            continue
        name = (s.name or "").strip()
        if not name:
            continue
        summ = (s.summary or "").strip()
        if summ and not _is_placeholder_summary(summ):
            known.add(name.lower())
    return known


def _needs_jit_for_term(store: MemoryStore, term: str) -> bool:
    rows = store.fetch_nodes_by_exact_names([term], limit=8)
    if not rows:
        return True
    best = rows[0]
    summ = str(best["summary"] or "").strip()
    return _is_placeholder_summary(summ)


def build_jit_web_knowledge_block(
    *,
    user_text: str,
    store: MemoryStore,
    graph_retriever: GraphRetriever,
    reasoning_result: ReasoningChainResult | None,
    persist: bool = True,
) -> str:
    """Return markdown-style lines for the prompt.

    When ``persist`` is True (default), upserts concept nodes and ``co_occurs`` edges
    to the detected topic head. Set ``persist=False`` for dry-run (e.g. prompt-debug).
    """
    if not jit_web_enabled():
        return ""

    strict = jit_web_strict_mode()
    if strict and not is_narrow_definition_question(user_text):
        return ""

    from src.RAG_online.internet_access import web_gloss_for_topic

    max_l = _max_lookups()
    if max_l == 0:
        return ""

    top_n = _scrape_top_n()
    known = _known_from_reasoning(reasoning_result)
    candidates = _collect_phrase_candidates(user_text, strict=strict)

    head_id, _head_name = graph_retriever.detect_head(user_text)

    lines: list[str] = []
    lookups = 0

    for cand in candidates:
        if lookups >= max_l:
            break
        ck = cand.lower()
        if ck in known:
            continue
        if not _needs_jit_for_term(store, cand):
            known.add(ck)
            continue

        try:
            gloss = web_gloss_for_topic(cand, top_n=top_n, max_bullets=5)
        except Exception as exc:
            _LOG.warning("jit web gloss failed for %r: %s", cand, exc)
            continue
        if not gloss:
            continue

        summary_body = gloss.strip()
        if len(summary_body) > 3900:
            summary_body = summary_body[:3897] + "..."

        if persist:
            try:
                node_id = store.upsert_node(
                    name=cand.strip(),
                    type_="concept",
                    summary=f"[web-jit] {summary_body}",
                )
            except Exception as exc:
                _LOG.warning("jit upsert_node failed for %r: %s", cand, exc)
                continue

            if head_id is not None:
                try:
                    store.upsert_edge(
                        src_id=int(node_id),
                        dst_id=int(head_id),
                        relation_type="co_occurs",
                        weight_delta=0.25,
                    )
                except Exception as exc:
                    _LOG.debug("jit edge link failed: %s", exc)

        known.add(ck)
        lookups += 1
        display = summary_body.replace("\n", " ")[:900]
        lines.append(f"- **{cand}** (web, unverified): {display}")

    if not lines:
        return ""

    return (
        "The following snippets were retrieved from the web for concepts that were not "
        "adequately defined in the long-term graph. Treat as unverified; prefer the user's "
        "facts when they conflict.\n" + "\n".join(lines)
    )

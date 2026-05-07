"""Deterministic intent classification for prompt framing (no LLM)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from src.memory_manager.retrieval.reasoning_chain import ReasoningChainResult


class IntentClassification(TypedDict):
    intent: str  # command | question | statement
    source: str


# Fixed lexicon: leading token (lowercase) after strip.
_IMPERATIVE_VERBS = frozenset(
    {
        "schedule",
        "plan",
        "draft",
        "remind",
        "add",
        "delete",
        "remove",
        "show",
        "list",
        "create",
        "send",
        "call",
        "book",
        "set",
        "update",
        "cancel",
        "start",
        "stop",
        "finish",
        "complete",
        "write",
        "make",
        "build",
        "run",
        "open",
        "close",
        "save",
        "email",
        "text",
        "tell",
        "give",
        "find",
        "get",
        "put",
        "move",
        "copy",
        "rename",
        "archive",
        "submit",
        "confirm",
        "approve",
        "reject",
    }
)

_DEFINITION_Q_RE = re.compile(r"\b(what|who|which)\b", re.IGNORECASE)


def is_narrow_definition_question(user_text: str) -> bool:
    """True when the message looks like a definitional question (JIT web gate).

    Requires a question mark and a *what*, *who*, or *which* cue to reduce false
    web lookups on casual chat.
    """
    t = (user_text or "").strip()
    if "?" not in t:
        return False
    return bool(_DEFINITION_Q_RE.search(t))


_INTERROGATIVE_PREFIXES = (
    "what ",
    "when ",
    "where ",
    "who ",
    "whom ",
    "whose ",
    "why ",
    "how ",
    "which ",
    "can ",
    "could ",
    "would ",
    "should ",
    "is ",
    "are ",
    "am ",
    "was ",
    "were ",
    "do ",
    "does ",
    "did ",
    "has ",
    "have ",
    "had ",
    "may ",
    "might ",
    "must ",
)


def classify_intent(
    user_text: str,
    _reasoning: ReasoningChainResult | None = None,
) -> IntentClassification:
    """Classify user message into command | question | statement (deterministic).

    `_reasoning` is reserved for future policy extensions; classification is text-first.
    """
    text = (user_text or "").strip()
    if not text:
        return {"intent": "statement", "source": "empty"}

    if text.startswith("/"):
        return {"intent": "command", "source": "slash"}

    if is_narrow_definition_question(text):
        return {"intent": "question", "source": "definition_question"}

    if "?" in text:
        return {"intent": "question", "source": "question_mark"}

    lower = text.lower()
    for prefix in _INTERROGATIVE_PREFIXES:
        if lower.startswith(prefix):
            return {"intent": "question", "source": "interrogative_prefix"}

    first = lower.split(maxsplit=1)[0] if lower else ""
    if first in _IMPERATIVE_VERBS:
        return {"intent": "command", "source": "imperative_lexicon"}

    return {"intent": "statement", "source": "default"}

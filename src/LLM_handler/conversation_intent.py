"""Discourse-aware interpretation using recent dialogue + current input (no LLM).

Resolves bare follow-ups such as ``explain?`` to the prior user turn and short
assistant reply so graph retrieval, memory, and JIT web can use grounded text
instead of an empty-looking current line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

_MAX_REASONING_INPUT_CHARS = 2800

# Short elaboration / continuation cues (whole message, after strip).
_RE_ELABORATION_FOLLOWUP = re.compile(
    r"(?is)^\s*("
    r"explain\??|"
    r"details?\??|"
    r"more(\s+info)?\??|"
    r"elaborate\b|"
    r"expand(\s+on\s+(that|it))?\??|"
    r"why(\s+so)?\??|"
    r"how\s+so\??|"
    r"tell\s+me\s+more\??|"
    r"go\s+on\??|"
    r"say\s+more\b"
    r")[\s?.!]*$"
)


def _role(msg: Mapping[str, str]) -> str:
    return (msg.get("role") or "").strip().lower()


def _last_user_then_assistant_pair(
    recent_messages: Iterable[Mapping[str, str]],
) -> tuple[str, str] | None:
    """If history ends with user then assistant, return (user_text, assistant_text)."""
    msgs = list(recent_messages)
    if len(msgs) < 2:
        return None
    u_msg, a_msg = msgs[-2], msgs[-1]
    if _role(u_msg) != "user" or _role(a_msg) != "assistant":
        return None
    u = (u_msg.get("content") or "").strip()
    a = (a_msg.get("content") or "").strip()
    if not u or not a:
        return None
    return u, a


def _assistant_reply_is_short(text: str, *, max_words: int = 32) -> bool:
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    return 0 < len(words) <= max_words


def _clip_for_reasoning(s: str, *, max_chars: int = _MAX_REASONING_INPUT_CHARS) -> str:
    t = s.strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 24].rstrip() + "\n…(truncated)"


@dataclass(frozen=True)
class DiscourseResolution:
    """Result of tying the current line to recent dialogue."""

    pattern: str
    """``default`` or ``elaboration_followup``."""

    reasoning_input: str
    """Text to feed graph/memory/JIT retrieval (usually the raw user line)."""

    summary: str
    """One-line description for logging or prompt metadata."""


def resolve_discourse_for_reasoning(
    user_input: str,
    recent_messages: Iterable[Mapping[str, str]],
) -> DiscourseResolution:
    """Return retrieval-oriented text when the current message is a bare follow-up.

    When the last two history messages are ``user`` then ``assistant``, the
    assistant reply is short, and the current input looks like a request to
    elaborate, we fuse prior user + assistant + current line so
    ``ReasoningChainEngine.build_chain`` (and callers that reuse this string for
    memory / JIT) see the topic the user originally raised.

    ``recent_messages`` should be the same ordered list used for
    ``[Recent conversation: …]`` (typically does **not** include the current
    user line yet).
    """
    raw = (user_input or "").strip()
    if not raw:
        return DiscourseResolution("default", raw, "")

    if not _RE_ELABORATION_FOLLOWUP.match(raw):
        return DiscourseResolution("default", raw, "")

    pair = _last_user_then_assistant_pair(recent_messages)
    if pair is None:
        return DiscourseResolution("default", raw, "")

    prev_user, last_assistant = pair
    if not _assistant_reply_is_short(last_assistant):
        return DiscourseResolution("default", raw, "")

    fused = (
        "Prior user message:\n"
        f"{prev_user}\n\n"
        "Assistant reply (brief):\n"
        f"{last_assistant}\n\n"
        "Current user follow-up:\n"
        f"{raw}\n"
    )
    clipped = _clip_for_reasoning(fused)
    return DiscourseResolution(
        pattern="elaboration_followup",
        reasoning_input=clipped,
        summary="Elaboration on prior user topic after short assistant reply.",
    )

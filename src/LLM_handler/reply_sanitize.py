"""Remove trailing echoes of internal prompt scaffolding from LLM replies."""

from __future__ import annotations

import re

# Bold/markdown "next action for user" footer (often redundant after a closing question).
_NEXT_ACTION_TAIL = re.compile(
    r"(?is)\n+\s*"
    r"\*{1,2}\s*"
    r"Next\s+(?:Action|Step)s?"  # Action, Actions, Step, Steps
    r"(?:\s+for\s+(?:the\s+)?User)?"
    r"\s*:\s*.+$"
)

# Lines the model often copies from the secretary prompt (reasoning steps, intent line).
# Model sometimes wraps spurious tags or stage directions in double dashes (e.g. "--emotion: …--").
_DOUBLE_DASH_SEGMENT = re.compile(r"--[\s\S]*?--")

_MARK_START = re.compile(
    r"(?is)(?:\n|^)\s*(?:"
    r"\[Concept\]"  # paraphrase of reasoning step format
    r"|\[Intent policy:"
    r"|\[Web knowledge"
    r"|\[Reasoning chain"
    r"|\[CLS-M memory:"
    r"|\[User input:"
    r"|\[Instruction:"
    r"|-\s*\[(?:concept|relation|evidence|command)\]"  # format_reasoning_block_text bullets
    r"|LINKED_CHAIN\b"
    r"|QUERY_TOKENS\b"
    r"|GRAPH_LINKS\b"
    r")",
)


def strip_internal_prompt_echo(text: str) -> str:
    """Drop trailing content from the first internal-looking bracket line onward."""
    if not text:
        return text
    m = _MARK_START.search(text)
    if m:
        return text[: m.start()].rstrip()
    return text.strip()


def strip_double_dash_segments(text: str) -> str:
    """Remove `-- ... --` blocks (often spurious tags / directions) and collapse whitespace."""
    if not text:
        return text
    t = _DOUBLE_DASH_SEGMENT.sub("", text)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def strip_next_action_boilerplate(text: str) -> str:
    """Remove trailing **Next Action for User:** … block (common redundant footer)."""
    if not text:
        return text
    m = _NEXT_ACTION_TAIL.search(text)
    if m:
        return text[: m.start()].rstrip()
    return text.strip()


def sanitize_assistant_reply(text: str) -> str:
    """Apply all reply cleanups (next-action footer, `--` segments, internal bracket echoes)."""
    t = strip_next_action_boilerplate(text)
    t = strip_double_dash_segments(t)
    return strip_internal_prompt_echo(t)

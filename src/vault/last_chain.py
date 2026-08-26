"""Process-local buffer of the last reasoning chain per session."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


class LastChainBuffer:
    """Keep a small LRU of ``ReasoningChainResult.to_dict()`` payloads."""

    def __init__(self, max_sessions: int = 48) -> None:
        self.max_sessions = max(1, int(max_sessions))
        self._data: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def put(self, session_id: str, payload: dict[str, Any]) -> None:
        sid = (session_id or "default").strip() or "default"
        self._data.pop(sid, None)
        self._data[sid] = payload
        while len(self._data) > self.max_sessions:
            self._data.popitem(last=False)

    def get(self, session_id: str) -> dict[str, Any] | None:
        sid = (session_id or "default").strip() or "default"
        item = self._data.get(sid)
        if item is None:
            return None
        self._data.move_to_end(sid)
        return item

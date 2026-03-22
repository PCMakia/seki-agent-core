from pathlib import Path
import json
import os
from datetime import datetime, timezone

LOG_DIR = Path(os.getenv("CHAT_LOG_DIR", "data/logs")).resolve()
JSON_LOG = LOG_DIR / "chat_log.jsonl"
TEXT_LOG = LOG_DIR / "chat_log.txt"

def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

def current_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def append_json_log(
    user_message: str,
    agent_reply: str,
    usage: dict | None = None,
    reasoning_meta: dict | None = None,
) -> None:
    _ensure_log_dir()
    record = {
        "timestamp": current_timestamp(),
        "user_message": user_message,
        "agent_reply": agent_reply,
    }
    if usage is not None:
        record["usage"] = usage
    if reasoning_meta is not None:
        record["reasoning_meta"] = reasoning_meta
    with JSON_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def append_text_log(user_message: str, agent_reply: str) -> None:
    _ensure_log_dir()
    with TEXT_LOG.open("a", encoding="utf-8") as f:
        f.write("Chat log:\n")
        f.write(f"User: {user_message}\n")
        f.write(f"Agent: {agent_reply}\n")
        f.write("\n")
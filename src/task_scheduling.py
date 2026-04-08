from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from src.llm_client import LLMClient
from src.memory_store import MemoryStore
from src.reminder_scheduler import ReminderItem, ReminderScheduler
from src.summarizer import bullet_summary_extractive
from src.time_utils import get_tz, now_dt, now_iso
from src.tools.windows.outlook_calendar import OutlookCalendarError, create_outlook_appointment

_LOG = logging.getLogger("personal_assistant.scheduling")


class ScheduledTaskPayload(BaseModel):
    subject: str = Field(..., min_length=1)
    body: str = ""
    # ISO-8601 with offset preferred; if no offset is present we interpret in app TZ.
    start: str
    duration_minutes: int = Field(60, ge=1, le=24 * 60)


@dataclass(frozen=True)
class ScheduleResult:
    created: bool
    task_id: str | None = None
    outlook_entry_id: str | None = None
    start_ts: str | None = None
    note: str = ""


EXTRACTION_PROMPT = """You extract calendar scheduling details from user text.

Return JSON ONLY that matches this schema:
{{
  "subject": "short title",
  "body": "optional details",
  "start": "ISO-8601 datetime (include timezone offset if possible)",
  "duration_minutes": 60
}}

Rules:
- Use the provided 'now' as reference for relative dates (today/tomorrow/next week).
- If user did not specify duration, use 60.
- If user text does not include a valid future time, still output JSON but choose the best guess; do NOT output nulls.

now={now_iso}
text={text}
"""


def _parse_start(start_str: str) -> datetime:
    raw = (start_str or "").strip()
    if not raw:
        raise ValueError("missing start")
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=get_tz())
    return dt


def _should_pre_summarize(user_text: str) -> bool:
    return len((user_text or "").strip()) >= 500


def _pre_summarize(user_text: str) -> str:
    # Keep this conservative to avoid dropping dates/times.
    return bullet_summary_extractive(user_text, max_bullets=10, ratio=0.6, max_words_per_bullet=60)


def extract_scheduled_task_payload(
    *,
    user_text: str,
    llm: LLMClient,
) -> ScheduledTaskPayload:
    raw = (user_text or "").strip()
    text_for_extraction = raw
    if _should_pre_summarize(raw):
        try:
            summarized = _pre_summarize(raw).strip()
            if summarized:
                text_for_extraction = summarized
        except Exception:
            text_for_extraction = raw

    prompt = EXTRACTION_PROMPT.format(now_iso=now_iso(), text=text_for_extraction)
    result = llm.generate(prompt, system_prompt="Return JSON only.", history=[])
    completion = (result.get("completion") or "").strip()
    try:
        data = json.loads(completion)
        return ScheduledTaskPayload.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        # Retry with raw text (avoid summary-induced field loss).
        prompt = EXTRACTION_PROMPT.format(now_iso=now_iso(), text=raw)
        result = llm.generate(prompt, system_prompt="Return JSON only.", history=[])
        completion = (result.get("completion") or "").strip()
        data = json.loads(completion)
        return ScheduledTaskPayload.model_validate(data)


async def schedule_from_user_input(
    *,
    session_id: str,
    user_text: str,
    llm: LLMClient,
    store: MemoryStore,
    scheduler: ReminderScheduler,
) -> ScheduleResult:
    sid = (session_id or "default").strip() or "default"
    raw = (user_text or "").strip()
    if not raw:
        return ScheduleResult(created=False, note="Empty input.")

    _LOG.info("[schedule] start session_id=%s user_text=%r", sid, raw[:200])
    payload = extract_scheduled_task_payload(user_text=raw, llm=llm)
    _LOG.info(
        "[schedule] extracted payload subject=%r start=%r duration_minutes=%s",
        payload.subject,
        payload.start,
        payload.duration_minutes,
    )
    start_dt = _parse_start(payload.start)
    duration_m = int(payload.duration_minutes)
    end_dt = start_dt + timedelta(minutes=duration_m)

    # Reminder policy: fire at start-30m; if already within 30m, fire ASAP.
    now = now_dt()
    delta = start_dt - now
    reminder_fire_at = start_dt - timedelta(minutes=30)
    if delta.total_seconds() <= 30 * 60:
        reminder_fire_at = now

    task_id = str(uuid.uuid4())
    tags = ["active", "task"]
    payload_json = payload.model_dump_json()
    tags_json = json.dumps(tags, ensure_ascii=False, separators=(",", ":"))
    start_ts = start_dt.isoformat(timespec="seconds")
    fire_ts = reminder_fire_at.isoformat(timespec="seconds")
    _LOG.info(
        "[schedule] computed times start_ts=%s reminder_fire_at=%s delta_seconds=%.1f",
        start_ts,
        fire_ts,
        delta.total_seconds(),
    )

    outlook_entry_id: str | None = None
    try:
        # Set Outlook reminder minutes so Outlook itself also reminds user.
        # If event is soon, reduce reminder lead time to avoid negative minutes.
        reminder_minutes = 30
        if delta.total_seconds() <= 30 * 60:
            reminder_minutes = max(0, int(delta.total_seconds() // 60))
        appt = create_outlook_appointment(
            subject=payload.subject,
            body=payload.body,
            start=start_dt,
            duration_minutes=duration_m,
            reminder_minutes_before_start=reminder_minutes,
        )
        outlook_entry_id = appt.entry_id
        _LOG.info("[schedule] calendar saved outlook_entry_id=%s", outlook_entry_id)
    except OutlookCalendarError as exc:
        # Fail-open: persist as unresolved so we can retry later.
        _LOG.warning("[schedule] calendar create failed; storing unresolved. error=%s", exc)
        store.upsert_unresolved(
            task_id=task_id,
            session_id=sid,
            instruction=raw,
            payload_json=payload_json,
            tags_json=tags_json,
            event_start_ts=start_ts,
            reminder_fire_at_ts=fire_ts,
            status="active",
        )
        await scheduler.add(
            ReminderItem(
                task_id=task_id,
                session_id=sid,
                instruction=raw,
                payload_json=payload_json,
                tags_json=tags_json,
                event_start_ts=start_ts,
                reminder_fire_at_ts=fire_ts,
            )
        )
        _LOG.info("[schedule] unresolved enqueued task_id=%s reminder_fire_at=%s", task_id, fire_ts)
        return ScheduleResult(
            created=False,
            task_id=task_id,
            start_ts=start_ts,
            note=str(exc),
        )

    # Persist only when deferred (>30m). Near-term reminders are covered by Outlook and the scheduler fire-at-now.
    if delta.total_seconds() > 30 * 60:
        _LOG.info("[schedule] deferred task (>30m): persisting unresolved + scheduler queue")
        store.upsert_unresolved(
            task_id=task_id,
            session_id=sid,
            instruction=raw,
            payload_json=payload_json,
            tags_json=tags_json,
            event_start_ts=start_ts,
            reminder_fire_at_ts=fire_ts,
            status="active",
        )
        await scheduler.add(
            ReminderItem(
                task_id=task_id,
                session_id=sid,
                instruction=raw,
                payload_json=payload_json,
                tags_json=tags_json,
                event_start_ts=start_ts,
                reminder_fire_at_ts=fire_ts,
            )
        )
        _LOG.info("[schedule] unresolved enqueued task_id=%s reminder_fire_at=%s", task_id, fire_ts)
    else:
        _LOG.info("[schedule] near-term task (<=30m): relying on calendar reminder + immediate scheduler timing")

    return ScheduleResult(
        created=True,
        task_id=task_id,
        outlook_entry_id=outlook_entry_id,
        start_ts=start_ts,
        note=f"Scheduled '{payload.subject}' from {start_ts} to {end_dt.isoformat(timespec='seconds')}.",
    )


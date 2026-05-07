""" Outlook interactive
    Script to create event in Classic Outlook (need bridge - in Execution_scripts - online)
"""
from __future__ import annotations

import json
import logging
import re
import sys
import uuid
import ast
import calendar
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from src.LLM_handler.llm_client import LLMClient
from src.memory_manager.storage.memory_store import MemoryStore
from src.reminder_scheduler import ReminderItem, ReminderScheduler
from src.RAG_online.summarizer import bullet_summary_extractive
from src.time_utils import get_tz, now_dt, now_iso
from src.tools_external.windows.outlook_calendar import (
    OutlookCalendarError,
    can_attempt_outlook_calendar,
    create_outlook_appointment,
)

_LOG = logging.getLogger("personal_assistant.scheduling")

# Heuristics: only on Windows; need a time/day cue and a calendar/scheduling cue.
_TIME_OR_CLOCK_RE = re.compile(
    r"(?:\b\d{1,2}:\d{2}\s*(?:[ap]\.?m\.?)?\b|\b\d{1,2}\s*[ap]\.?\s*m\.?\b)",
    re.IGNORECASE,
)
_REL_OR_NAMED_DATE_RE = re.compile(
    r"\b(?:"
    r"today|tomorrow|tonight|"
    r"this\s+(?:mon|tues|wednes|thurs|fri|satur|sun)day|"
    r"next\s+(?:mon|tues|wednes|thurs|fri|satur|sun)day|"
    r"next\s+week|this\s+week|"
    r"(?:mon|tues|wednes|thurs|fri|satur|sun)day|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}"
    r")\b",
    re.IGNORECASE,
)
_CAL_ACTION_RE = re.compile(
    r"(?:"
    r"\bwindows?\s+calendar\b|"
    r"\bmicrosoft\s+outlook\b|"
    r"\boutlook(?:\s+calendar)?\b|"
    r"\bremind(?:\s+me)?\b|"
    r"\badd\s+(?:an?\s+)?(?:to\s+)?(?:my\s+)?(?:outlook\s+)?(?:the\s+)?(?:a\s+)?(?:event|appointment|meeting|reminder)\b|"
    r"\badd\s+to\s+(?:my\s+|the\s+)?(?:outlook\s+)?calendar\b|"
    r"\bput\s+on\s+(?:my\s+)?(?:outlook\s+)?calendar\b|"
    r"\bschedule(?:d|s|ing)?\b|"
    r"\bcalendar\b|"
    r"\bappointment\b|"
    r"\bbook(?:ing)?\b"
    r")",
    re.IGNORECASE,
)

_SCHEDULE_TAG_PREFIX = "/schedule"


def _strip_schedule_tag_prefix(user_text: str) -> str:
    raw = (user_text or "").strip()
    if not raw:
        return ""
    low = raw.lower()
    if not low.startswith(_SCHEDULE_TAG_PREFIX):
        return raw
    rest = raw[len(_SCHEDULE_TAG_PREFIX) :].lstrip(" :,-")
    return rest.strip() or raw


def should_attempt_windows_calendar_schedule(user_text: str) -> bool:
    """True when the text likely requests a Windows/Outlook calendar action with a concrete time or day.

    Kept deliberately separate from the main LLM prompt: this gate only triggers
    :func:`schedule_from_user_input`, which uses a small extraction prompt and
    :func:`src.tools.windows.outlook_calendar.create_outlook_appointment`.
    """
    raw = (user_text or "").strip()
    if not raw or not can_attempt_outlook_calendar():
        return False
    if raw.lower().startswith(_SCHEDULE_TAG_PREFIX):
        return True
    normalized = _strip_schedule_tag_prefix(raw)
    if not (_TIME_OR_CLOCK_RE.search(normalized) or _REL_OR_NAMED_DATE_RE.search(normalized)):
        return False
    if _CAL_ACTION_RE.search(normalized):
        return True
    low = normalized.lower()
    if re.search(r"\bwindows?\b", low) and re.search(
        r"\b(?:calendar|outlook|schedule|meeting|event|appointment)\b", low
    ):
        return True
    return False


class ScheduledTaskPayload(BaseModel):
    subject: str = Field(..., min_length=1)
    body: str = ""
    # ISO-8601 with offset preferred; if no offset is present we interpret in app TZ.
    start: str
    duration_minutes: int = Field(60, ge=1, le=24 * 60)

    @classmethod
    def model_validate(cls, obj: Any, *args: Any, **kwargs: Any):  # type: ignore[override]
        if isinstance(obj, dict) and obj.get("body") is None:
            obj = dict(obj)
            obj["body"] = ""
        return super().model_validate(obj, *args, **kwargs)


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
    try:
        dt = datetime.fromisoformat(raw)
    except Exception:
        dt = _parse_noisy_datetime_hint(raw)
    app_tz = _runtime_local_tz()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=app_tz)
    # Normalize all tz-aware values to app timezone so local wall-clock
    # scheduling remains consistent (e.g., "3pm" should remain 15:00 local).
    return dt.astimezone(app_tz)


def _parse_noisy_datetime_hint(raw: str) -> datetime:
    """Recover useful datetime values from non-ISO LLM chatter."""
    text = (raw or "").strip().strip("\"'")
    lower = text.lower()
    end_of_day = "end of day" in lower or "eod" in lower

    # 1) Extract a full datetime first if present.
    full_dt = re.search(
        r"(\d{4}-\d{2}-\d{2}[t\s]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:z|[+-]\d{2}:?\d{2})?)",
        text,
        re.IGNORECASE,
    )
    if full_dt:
        candidate = full_dt.group(1).replace("Z", "+00:00").replace("z", "+00:00")
        return datetime.fromisoformat(candidate)

    # 2) Date-only with day.
    ymd = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if ymd:
        year, month, day = int(ymd.group(1)), int(ymd.group(2)), int(ymd.group(3))
        hour, minute = (23, 59) if end_of_day else (9, 0)
        return datetime(year, month, day, hour, minute, 0)

    # 3) Year-month only (e.g. 2026-05 end of day) -> infer day boundary.
    ym = re.search(r"\b(\d{4})-(\d{2})\b", text)
    if ym:
        year, month = int(ym.group(1)), int(ym.group(2))
        if not 1 <= month <= 12:
            raise ValueError(f"Invalid month in noisy datetime: {raw!r}")
        day = calendar.monthrange(year, month)[1] if end_of_day else 1
        hour, minute = (23, 59) if end_of_day else (9, 0)
        return datetime(year, month, day, hour, minute, 0)

    raise ValueError(f"Invalid isoformat string: {raw!r}")


def _runtime_local_tz():
    """Prefer host/runtime local timezone"""
    try:
        tz = datetime.now().astimezone().tzinfo
        if tz is not None:
            return tz
    except Exception:
        pass
    return get_tz()


_WEEKDAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_WEEKDAY_RE = re.compile(
    r"\b(?:(next|this)\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_TIME_12H_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s*m\.?\b", re.IGNORECASE)
_TIME_24H_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_DURATION_RE = re.compile(r"\bfor\s+(\d{1,3})\s*(?:minutes?|mins?|m)\b", re.IGNORECASE)


def _extract_duration_minutes_from_text(user_text: str) -> int | None:
    m = _DURATION_RE.search(user_text or "")
    if not m:
        return None
    try:
        val = int(m.group(1))
        if 1 <= val <= 24 * 60:
            return val
    except Exception:
        return None
    return None


def _extract_time_from_text(user_text: str) -> tuple[int, int] | None:
    txt = user_text or ""
    m12 = _TIME_12H_RE.search(txt)
    if m12:
        hour = int(m12.group(1))
        minute = int(m12.group(2) or "0")
        ap = m12.group(3).lower()
        if hour == 12:
            hour = 0
        if ap == "p":
            hour += 12
        return hour, minute
    m24 = _TIME_24H_RE.search(txt)
    if m24:
        return int(m24.group(1)), int(m24.group(2))
    return None


def _has_explicit_date_cue(user_text: str) -> bool:
    low = (user_text or "").lower()
    if any(k in low for k in ("today", "tomorrow", "tonight", "next ", "this ")):
        return True
    return bool(_WEEKDAY_RE.search(low))


def _extract_date_from_text(user_text: str, base: datetime) -> datetime:
    low = (user_text or "").lower()
    current = base.date()
    if "tomorrow" in low:
        return base + timedelta(days=1)
    if "today" in low or "tonight" in low:
        return base

    wm = _WEEKDAY_RE.search(low)
    if wm:
        qual = (wm.group(1) or "").lower()
        wd_name = wm.group(2).lower()
        target = _WEEKDAY_INDEX[wd_name]
        current_wd = current.weekday()
        delta = (target - current_wd) % 7
        if qual == "next":
            delta = 7 if delta == 0 else delta
        elif qual == "this":
            # "this Wednesday" should be in current week; if already passed, roll one week.
            if delta == 0:
                delta = 0
        else:
            # Bare weekday means next occurrence (today allowed).
            delta = delta
        return base + timedelta(days=delta)

    return base


def _parse_start_from_user_text(user_text: str) -> datetime:
    """Deterministic fallback when LLM returns invalid/partial datetime."""
    now = now_dt()
    date_base = _extract_date_from_text(user_text, now)
    hm = _extract_time_from_text(user_text) or (9, 0)
    dt = date_base.replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)
    # If no explicit date cue and chosen time already passed, schedule next day.
    low = (user_text or "").lower()
    has_date_cue = any(k in low for k in ("today", "tomorrow", "tonight", "next ", "this ")) or bool(
        _WEEKDAY_RE.search(low)
    )
    if not has_date_cue and dt <= now:
        dt = dt + timedelta(days=1)
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
        data = _parse_llm_schedule_payload(completion)
        return ScheduledTaskPayload.model_validate(data)
    except (json.JSONDecodeError, ValidationError, ValueError, SyntaxError):
        # Retry with raw text (avoid summary-induced field loss).
        prompt = EXTRACTION_PROMPT.format(now_iso=now_iso(), text=raw)
        result = llm.generate(prompt, system_prompt="Return JSON only.", history=[])
        completion = (result.get("completion") or "").strip()
        data = _parse_llm_schedule_payload(completion)
        return ScheduledTaskPayload.model_validate(data)


def _parse_llm_schedule_payload(completion: str) -> dict[str, Any]:
    """Parse schedule payload from imperfect LLM output.

    Accepts strict JSON, fenced JSON blocks, and Python-dict style payloads.
    """
    text = (completion or "").strip()
    if not text:
        raise ValueError("Empty extraction payload")

    # 1) Fast path: strict JSON body.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 2) Remove markdown code fences and retry JSON.
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence_match:
        fenced = fence_match.group(1).strip()
        try:
            parsed = json.loads(fenced)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            text = fenced

    # 3) Extract first {...} block if there is surrounding prose.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1].strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            # 4) Last-resort: Python dict literal (single quotes, etc.).
            try:
                lit = ast.literal_eval(candidate)
                if isinstance(lit, dict):
                    return lit
            except Exception:
                pass

    raise ValueError("Could not parse schedule payload as JSON/dict")


async def schedule_from_user_input(
    *,
    session_id: str,
    user_text: str,
    llm: LLMClient,
    store: MemoryStore,
    scheduler: ReminderScheduler,
) -> ScheduleResult:
    sid = (session_id or "default").strip() or "default"
    raw = _strip_schedule_tag_prefix(user_text)
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
    try:
        start_dt = _parse_start(payload.start)
    except Exception as exc:
        _LOG.warning("[schedule] invalid extracted start=%r; falling back to user-text parser: %s", payload.start, exc)
        start_dt = _parse_start_from_user_text(raw)

    # Guardrail: if user provided an explicit clock time (e.g. 3pm), trust that
    # over LLM-derived timezone/time drift.
    explicit_hm = _extract_time_from_text(raw)
    if explicit_hm is not None:
        anchor = start_dt
        if _has_explicit_date_cue(raw):
            anchor = _extract_date_from_text(raw, now_dt())
        start_dt = start_dt.replace(
            year=anchor.year,
            month=anchor.month,
            day=anchor.day,
            hour=explicit_hm[0],
            minute=explicit_hm[1],
            second=0,
            microsecond=0,
        )

    duration_m = int(payload.duration_minutes)
    duration_from_text = _extract_duration_minutes_from_text(raw)
    if duration_from_text is not None:
        duration_m = duration_from_text
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
        # Temporary offset correction for Outlook write path:
        # shift parsed time 4 hours earlier before creating the event.
        start_for_outlook = start_dt - timedelta(hours=4)
        # Outlook COM works best with local wall-clock datetimes (naive), not
        # offset-aware timestamps that can be reinterpreted by calendar timezone.
        start_local_wall_clock = start_for_outlook.replace(tzinfo=None)
        _LOG.info(
            "[schedule] outlook shift applied parsed_start=%s shifted_start=%s",
            start_dt.isoformat(timespec="seconds"),
            start_for_outlook.isoformat(timespec="seconds"),
        )
        appt = create_outlook_appointment(
            subject=payload.subject,
            body=payload.body,
            start=start_local_wall_clock,
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


async def outlook_calendar_side_effect_block(
    *,
    session_id: str,
    user_text: str,
    llm: LLMClient,
    store: MemoryStore,
    scheduler: ReminderScheduler,
) -> str:
    """If the user text looks like a Windows/Outlook calendar add, run scheduling and return a context line for the secretary prompt.

    Uses the same path as ``/agent/schedule`` (LLM JSON extraction + ``create_outlook_appointment``); does nothing when the heuristic gate is closed.
    """
    if not should_attempt_windows_calendar_schedule(user_text):
        return ""
    try:
        res = await schedule_from_user_input(
            session_id=session_id,
            user_text=user_text,
            llm=llm,
            store=store,
            scheduler=scheduler,
        )
    except Exception as exc:
        _LOG.warning("[schedule] auto gate: failed: %s", exc)
        return f"System (Windows calendar): scheduling could not run: {exc}"
    if res.created and (res.note or res.outlook_entry_id or res.start_ts):
        return f"System (Windows/Outlook calendar): {res.note or 'Event was written to the local calendar.'}"
    if res.note:
        return f"System (Windows/Outlook calendar): not created. {res.note}"
    return ""


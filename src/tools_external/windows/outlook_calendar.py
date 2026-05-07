from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class OutlookAppointmentResult:
    entry_id: str | None
    subject: str
    start: datetime
    duration_minutes: int


class OutlookCalendarError(RuntimeError):
    pass


def can_attempt_outlook_calendar() -> bool:
    """Return True when runtime can attempt Outlook calendar writes."""
    if os.getenv("OUTLOOK_BRIDGE_URL", "").strip():
        return True
    if sys.platform.startswith("win"):
        return True
    return bool(_powershell_candidates())


def _normalize_inputs(
    *,
    subject: str,
    start: datetime,
    duration_minutes: int,
    reminder_minutes_before_start: int,
) -> tuple[str, int, int]:
    subj = (subject or "").strip()
    if not subj:
        raise OutlookCalendarError("Missing required field: subject")
    if not isinstance(start, datetime):
        raise OutlookCalendarError("Missing/invalid required field: start datetime")

    mins = max(1, int(duration_minutes))
    rem = max(0, int(reminder_minutes_before_start))
    return subj, mins, rem


def _create_outlook_appointment_windows_native(
    *,
    subject: str,
    body: str,
    start: datetime,
    duration_minutes: int,
    reminder_minutes_before_start: int,
) -> OutlookAppointmentResult:
    # COM initialization is required when called from worker threads.
    import pythoncom  # type: ignore

    pythoncom.CoInitialize()
    try:
        import win32com.client  # type: ignore

        outlook = win32com.client.Dispatch("Outlook.Application")
        appointment = outlook.CreateItem(1)  # 1 = AppointmentItem
        appointment.Subject = subject
        appointment.Body = (body or "").strip()
        appointment.Start = start
        appointment.Duration = duration_minutes
        appointment.ReminderSet = True
        appointment.ReminderMinutesBeforeStart = reminder_minutes_before_start
        appointment.Save()
        entry_id = None
        try:
            entry_id = str(getattr(appointment, "EntryID", "") or "") or None
        except Exception:
            entry_id = None
        return OutlookAppointmentResult(
            entry_id=entry_id,
            subject=subject,
            start=start,
            duration_minutes=duration_minutes,
        )
    finally:
        pythoncom.CoUninitialize()


def _powershell_candidates() -> list[str]:
    candidates = [
        os.getenv("OUTLOOK_BRIDGE_POWERSHELL", "").strip(),
        "powershell.exe",
        "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
    ]
    out: list[str] = []
    for c in candidates:
        if not c:
            continue
        if "/" in c or "\\" in c:
            if os.path.exists(c):
                out.append(c)
            continue
        resolved = shutil.which(c)
        if resolved:
            out.append(resolved)
    return out


def _create_outlook_appointment_powershell_bridge(
    *,
    subject: str,
    body: str,
    start: datetime,
    duration_minutes: int,
    reminder_minutes_before_start: int,
) -> OutlookAppointmentResult:
    powershell_bins = _powershell_candidates()
    if not powershell_bins:
        raise OutlookCalendarError(
            "Outlook bridge unavailable from Linux/WSL runtime: could not find powershell.exe. "
            "Set OUTLOOK_BRIDGE_POWERSHELL to the Windows PowerShell executable path."
        )

    ps_script = (
        "$ErrorActionPreference = 'Stop'; "
        "$subject = $env:OA_SUBJECT; "
        "$body = $env:OA_BODY; "
        "$startIso = $env:OA_START_ISO; "
        "$duration = [int]$env:OA_DURATION_MINUTES; "
        "$rem = [int]$env:OA_REMINDER_MINUTES; "
        # Keep the parsed wall-clock time from the payload; avoid timezone conversion drift.
        "$startLocal = [DateTimeOffset]::Parse($startIso).DateTime; "
        "$outlook = New-Object -ComObject Outlook.Application; "
        "$appt = $outlook.CreateItem(1); "
        "$appt.Subject = $subject; "
        "$appt.Body = $body; "
        "$appt.Start = $startLocal; "
        "$appt.Duration = $duration; "
        "$appt.ReminderSet = $true; "
        "$appt.ReminderMinutesBeforeStart = $rem; "
        "$appt.Save(); "
        "$eid = ''; "
        "try { $eid = [string]$appt.EntryID } catch { $eid = '' }; "
        "$result = @{ entry_id = $eid; subject = $subject }; "
        "$result | ConvertTo-Json -Compress"
    )

    env = os.environ.copy()
    env["OA_SUBJECT"] = subject
    env["OA_BODY"] = (body or "").strip()
    env["OA_START_ISO"] = start.isoformat()
    env["OA_DURATION_MINUTES"] = str(duration_minutes)
    env["OA_REMINDER_MINUTES"] = str(reminder_minutes_before_start)

    last_error: Exception | None = None
    for ps_bin in powershell_bins:
        try:
            proc = subprocess.run(
                [ps_bin, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
                capture_output=True,
                text=True,
                env=env,
                timeout=float(os.getenv("OUTLOOK_BRIDGE_TIMEOUT_SEC", "25")),
                check=False,
            )
            if proc.returncode != 0:
                raise OutlookCalendarError(
                    f"PowerShell bridge failed (exit={proc.returncode}): {(proc.stderr or proc.stdout).strip()[:900]}"
                )
            raw = (proc.stdout or "").strip()
            if not raw:
                raise OutlookCalendarError("PowerShell bridge returned empty output.")
            # Some PowerShell hosts emit multiple lines; parse the last JSON-looking line.
            json_line = raw.splitlines()[-1].strip()
            data = json.loads(json_line)
            return OutlookAppointmentResult(
                entry_id=str(data.get("entry_id") or "").strip() or None,
                subject=subject,
                start=start,
                duration_minutes=duration_minutes,
            )
        except Exception as exc:
            last_error = exc
            continue
    raise OutlookCalendarError(
        "Failed to create Outlook event via PowerShell bridge from Linux/WSL runtime. "
        f"Last error: {last_error}"
    )


def _create_outlook_appointment_http_bridge(
    *,
    subject: str,
    body: str,
    start: datetime,
    duration_minutes: int,
    reminder_minutes_before_start: int,
) -> OutlookAppointmentResult:
    base = os.getenv("OUTLOOK_BRIDGE_URL", "").strip().rstrip("/")
    if not base:
        raise OutlookCalendarError("OUTLOOK_BRIDGE_URL is not configured.")
    url = f"{base}/create_event"
    payload = {
        "subject": subject,
        "body": (body or "").strip(),
        "start_iso": start.isoformat(),
        "duration_minutes": int(duration_minutes),
        "reminder_minutes_before_start": int(reminder_minutes_before_start),
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout_s = float(os.getenv("OUTLOOK_BRIDGE_TIMEOUT_SEC", "25"))
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = (resp.read() or b"").decode("utf-8", errors="replace").strip()
            if not raw:
                raise OutlookCalendarError("HTTP bridge returned empty response.")
            out = json.loads(raw)
    except Exception as exc:
        msg = str(exc)
        if "Connection refused" in msg or "[Errno 111]" in msg or "WinError 10061" in msg:
            raise OutlookCalendarError(
                f"HTTP bridge request failed ({url}): {exc}. "
                "The Outlook bridge service is not reachable from Docker. "
                "Start the host bridge server (for example, run `python Execution_scripts/outlook_bridge_server.py` on the host) "
                "or set `OUTLOOK_BRIDGE_URL` to the correct reachable endpoint."
            ) from exc
        raise OutlookCalendarError(f"HTTP bridge request failed ({url}): {exc}") from exc
    if not isinstance(out, dict):
        raise OutlookCalendarError("HTTP bridge returned unexpected payload.")
    if not bool(out.get("ok", False)):
        raise OutlookCalendarError(str(out.get("error") or "HTTP bridge failed"))
    entry_id = str(out.get("entry_id") or "").strip() or None
    start_com = str(out.get("start_com") or out.get("start_saved_utc") or "").strip()
    start_local_hint = str(
        out.get("start_local_hint") or out.get("start_saved_local") or ""
    ).strip()
    ol_tz = str(out.get("outlook_timezone_id") or "").strip()
    if start_com or start_local_hint:
        print(
            "[outlook bridge] outlook_tz=%s start_com=%s start_local_hint=%s"
            % (ol_tz or "(default)", start_com, start_local_hint)
        )
    return OutlookAppointmentResult(
        entry_id=entry_id,
        subject=subject,
        start=start,
        duration_minutes=duration_minutes,
    )


def create_outlook_appointment(
    *,
    subject: str,
    body: str = "",
    start: datetime,
    duration_minutes: int = 60,
    reminder_minutes_before_start: int = 15,
) -> OutlookAppointmentResult:
    """Create and save an Outlook Calendar appointment.

    On native Windows, uses in-process COM automation.
    On Linux/WSL runtimes, uses a PowerShell bridge to Windows host COM.

    Permission/session caveats (common failure modes on both paths):
    - Outlook must be installed, and the current user session must allow COM automation.
    - Some corporate policies block programmatic access; catch exceptions and show guidance.
    """
    subj, mins, rem = _normalize_inputs(
        subject=subject,
        start=start,
        duration_minutes=duration_minutes,
        reminder_minutes_before_start=reminder_minutes_before_start,
    )

    try:
        # Preferred cross-boundary mode for Docker/WSL: call host bridge service.
        if os.getenv("OUTLOOK_BRIDGE_URL", "").strip():
            return _create_outlook_appointment_http_bridge(
                subject=subj,
                body=body,
                start=start,
                duration_minutes=mins,
                reminder_minutes_before_start=rem,
            )
        if sys.platform.startswith("win"):
            return _create_outlook_appointment_windows_native(
                subject=subj,
                body=body,
                start=start,
                duration_minutes=mins,
                reminder_minutes_before_start=rem,
            )
        return _create_outlook_appointment_powershell_bridge(
            subject=subj,
            body=body,
            start=start,
            duration_minutes=mins,
            reminder_minutes_before_start=rem,
        )
    except OutlookCalendarError:
        raise
    except Exception as exc:
        raise OutlookCalendarError(
            "Failed to create Outlook calendar event. "
            "Check that Outlook is installed and available in the current Windows user session, "
            "and that programmatic access is allowed by your Office security settings/policies. "
            f"Underlying error: {exc}"
        ) from exc

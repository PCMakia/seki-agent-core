from __future__ import annotations

import json
import os
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

# Default bridge timezone (EDT/EST) when env var is not provided.
DEFAULT_BRIDGE_TZ = "America/New_York"

# Outlook COM uses Windows registry timezone IDs, not IANA. Map common cases;
# override with OUTLOOK_BRIDGE_OUTLOOK_TZ_ID (e.g. "Eastern Standard Time").
IANA_TO_OUTLOOK_TIMEZONE_ID: dict[str, str] = {
    "America/New_York": "Eastern Standard Time",
    "America/Detroit": "Eastern Standard Time",
    "America/Chicago": "Central Standard Time",
    "America/Denver": "Mountain Standard Time",
    "America/Los_Angeles": "Pacific Standard Time",
    "America/Phoenix": "US Mountain Standard Time",
    "America/Anchorage": "Alaskan Standard Time",
    "Pacific/Honolulu": "Hawaiian Standard Time",
    "America/Indiana/Indianapolis": "US Eastern Standard Time",
    "America/Indiana/Knox": "Central Standard Time",
    "America/Indiana/Vevay": "US Eastern Standard Time",
}


def _outlook_timezone_id_for_iana(iana: str) -> str:
    explicit = str(os.getenv("OUTLOOK_BRIDGE_OUTLOOK_TZ_ID", "")).strip()
    if explicit:
        return explicit
    return IANA_TO_OUTLOOK_TIMEZONE_ID.get(iana, "Eastern Standard Time")


def _apply_outlook_start_timezone(appointment: object, outlook: object, iana_tz: str) -> bool:
    """Set StartTimeZone/EndTimeZone for how the appointment displays in the calendar grid."""
    tz_id = _outlook_timezone_id_for_iana(iana_tz)
    try:
        tzs = outlook.TimeZones
        ol_tz = tzs.Item(tz_id)
    except Exception as exc:
        print(f"[outlook-bridge] WARNING: could not resolve Outlook.TimeZones.Item({tz_id!r}): {exc}")
        return False
    try:
        appointment.StartTimeZone = ol_tz
        appointment.EndTimeZone = ol_tz
        return True
    except Exception as exc:
        print(f"[outlook-bridge] WARNING: could not set StartTimeZone/EndTimeZone: {exc}")
        return False


def _create_outlook_event(
    *,
    subject: str,
    body: str,
    start_iso: str,
    duration_minutes: int,
    reminder_minutes_before_start: int,
) -> tuple[str | None, str, str]:
    import pythoncom  # type: ignore
    import win32com.client  # type: ignore

    target_tz_name = str(os.getenv("OUTLOOK_BRIDGE_TARGET_TZ", DEFAULT_BRIDGE_TZ)).strip() or DEFAULT_BRIDGE_TZ
    start = datetime.fromisoformat((start_iso or "").strip())
    subj = (subject or "").strip()
    if not subj:
        raise ValueError("Missing required field: subject")

    mins = max(1, int(duration_minutes))
    rem = max(0, int(reminder_minutes_before_start))
    # Temporary behavior: keep incoming wall-clock as-is and avoid timezone shifts.
    # If input contains tzinfo, drop it and write the same local clock values.
    start_for_outlook = start.replace(tzinfo=None)
    start_intended_local = start_for_outlook.isoformat(timespec="seconds")

    pythoncom.CoInitialize()
    try:
        outlook = win32com.client.Dispatch("Outlook.Application")
        appointment = outlook.CreateItem(1)  # AppointmentItem
        appointment.Subject = subj
        appointment.Body = (body or "").strip()
        # Temporary behavior: do not set StartTimeZone/EndTimeZone.
        tz_ok = False
        # Write local wall-clock scheduling via Start + Duration.
        appointment.Start = start_for_outlook
        appointment.Duration = mins
        print(
            "[outlook-bridge] write_start start_for_outlook=%s (type=%s) source_start=%s target_tz=%s"
            % (
                start_for_outlook.isoformat(),
                type(start_for_outlook).__name__,
                start.isoformat(),
                target_tz_name,
            )
        )
        print(
            "[outlook-bridge] set Start=%s Duration=%sm (local-only, iana=%s outlook_tz=%s tz_ok=%s)"
            % (
                start_for_outlook.isoformat(),
                mins,
                target_tz_name,
                _outlook_timezone_id_for_iana(target_tz_name),
                tz_ok,
            )
        )
        appointment.ReminderSet = True
        appointment.ReminderMinutesBeforeStart = rem
        appointment.Save()
        try:
            entry_id = str(getattr(appointment, "EntryID", "") or "") or None
        except Exception:
            entry_id = None
        hint = "intended_local=%s" % start_intended_local
        return entry_id, start_intended_local, hint
    finally:
        pythoncom.CoUninitialize()


class OutlookBridgeHandler(BaseHTTPRequestHandler):
    def _write_json(self, code: int, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/health":
            self._write_json(200, {"ok": True, "service": "outlook-bridge"})
            return
        self._write_json(404, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/create_event":
            self._write_json(404, {"ok": False, "error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b""
            data = json.loads(raw.decode("utf-8")) if raw else {}
            entry_id, start_intended_local, start_local_hint = _create_outlook_event(
                subject=str(data.get("subject", "")),
                body=str(data.get("body", "")),
                start_iso=str(data.get("start_iso", "")),
                duration_minutes=int(data.get("duration_minutes", 60)),
                reminder_minutes_before_start=int(data.get("reminder_minutes_before_start", 15)),
            )
            print(
                "[outlook-bridge] iana=%s outlook_tz=%s saved entry_id=%s start_com=%s start_local_hint=%s"
                % (
                    str(os.getenv("OUTLOOK_BRIDGE_TARGET_TZ", DEFAULT_BRIDGE_TZ)),
                    _outlook_timezone_id_for_iana(str(os.getenv("OUTLOOK_BRIDGE_TARGET_TZ", DEFAULT_BRIDGE_TZ))),
                    entry_id,
                    start_intended_local,
                    start_local_hint,
                )
            )
            self._write_json(
                200,
                {
                    "ok": True,
                    "entry_id": entry_id,
                    "start_com": start_intended_local,
                    "start_local_hint": start_local_hint,
                    "outlook_timezone_id": _outlook_timezone_id_for_iana(
                        str(os.getenv("OUTLOOK_BRIDGE_TARGET_TZ", DEFAULT_BRIDGE_TZ))
                    ),
                },
            )
        except Exception as exc:
            self._write_json(400, {"ok": False, "error": str(exc)})


def main() -> None:
    os.environ.setdefault("OUTLOOK_BRIDGE_TARGET_TZ", DEFAULT_BRIDGE_TZ)
    host = "0.0.0.0"
    port = 8765
    server = HTTPServer((host, port), OutlookBridgeHandler)
    print(
        "Outlook bridge listening on http://%s:%s (tz=%s)"
        % (host, port, os.getenv("OUTLOOK_BRIDGE_TARGET_TZ", DEFAULT_BRIDGE_TZ))
    )
    server.serve_forever()


if __name__ == "__main__":
    main()


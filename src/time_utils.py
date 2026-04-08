from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

# Single source of truth for local time interpretation across:
# - prompt time context
# - scheduling math
# - episode timestamps (per project plan)
TIMEZONE_NAME: str = str(os.getenv("APP_TIMEZONE", "America/New_York")).strip() or "America/New_York"


def get_tz() -> ZoneInfo:
    return ZoneInfo(TIMEZONE_NAME)


def now_dt() -> datetime:
    """Return timezone-aware 'now' in the configured app timezone."""
    return datetime.now(get_tz())


def now_iso(*, timespec: str = "seconds") -> str:
    """Return ISO-8601 timestamp in the configured app timezone."""
    return now_dt().isoformat(timespec=timespec)


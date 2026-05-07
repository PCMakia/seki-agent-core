from __future__ import annotations

import subprocess
import sys


def notify_windows_reminder(*, title: str, message: str) -> bool:
    """Show a small Windows toast-style reminder (best-effort).

    Returns True if a notification command was launched, False otherwise.
    """
    if not sys.platform.startswith("win"):
        return False

    t = (title or "Reminder").strip() or "Reminder"
    m = (message or "").strip() or "You have an upcoming scheduled task."

    # Escape single quotes for PowerShell single-quoted strings.
    t = t.replace("'", "''")
    m = m.replace("'", "''")

    script = (
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null;"
        "$template = [Windows.UI.Notifications.ToastTemplateType]::ToastText02;"
        "$xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent($template);"
        "$texts = $xml.GetElementsByTagName('text');"
        f"$texts.Item(0).AppendChild($xml.CreateTextNode('{t}')) > $null;"
        f"$texts.Item(1).AppendChild($xml.CreateTextNode('{m}')) > $null;"
        "$toast = [Windows.UI.Notifications.ToastNotification]::new($xml);"
        "$notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('PersonalAssistant');"
        "$notifier.Show($toast);"
    )

    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


import time
import sys

CLEAR = "\033[2]"
CLEAR_AND_RETURN = "\033[H"


def _play_alarm_sound() -> None:
    # First choice: playsound if available.
    try:
        from playsound import playsound  # type: ignore

        playsound("Alarm.mp3")
        return
    except Exception:
        pass

    # Fallback on Windows with no external dependency.
    if sys.platform.startswith("win"):
        try:
            import winsound

            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            return
        except Exception:
            pass

    # Last fallback: visible terminal signal.
    print("\n[alarm] Reminder time reached.")

def alarm(seconds):
    time_elapsed = 0

    print(CLEAR)
    while time_elapsed < seconds:
        time.sleep(1)
        time_elapsed += 1
        
        time_left = seconds - time_elapsed
        minutes_left = time_left // 60
        seconds_left = time_left % 60

        print(f"\rTime remaining: {minutes_left:02d}:{seconds_left:02d}", end="")

    _play_alarm_sound()

if __name__ == "__main__":
    alarm(10)
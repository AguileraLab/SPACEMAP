"""One format for everything the interface prints to the terminal.

Every line carries the time, the tab and, where useful, the step inside the tab,
so a long run can be followed from the terminal alone::

    2026-09-22 14:03:11 | Tab 7 Phenotyping | Resolve | Threshold mode: Z1 score
    2026-09-22 14:03:11 | Tab 7 Phenotyping | Manual Reference | Started: load columns
    2026-09-22 14:03:14 | Tab 7 Phenotyping | Manual Reference | Finished: load columns (3.0 s)

Which messages appear is controlled by environment variables:

``SPACEMAP_CONSOLE_LOGS``  master switch; ``1`` shows everything, ``0`` nothing
``SPACEMAP_TAB_LOGS``      pipeline-tab progress (on by default)
``SPACEMAP_PERF_LOGS``     page timing (on by default)
``SPACEMAP_APP_LOGS``      app start-up and project-file saves (off by default)
"""

from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager
from datetime import datetime

TAB_LABELS = {
    "app": "SPACEMAP",
    "project": "Project file",
    "tab0": "Tab 0 Project",
    "tab1": "Tab 1 Registration",
    "tab2": "Tab 2 Background",
    "tab3": "Tab 3 ROI Extraction",
    "tab4": "Tab 4 Annotation",
    "tab5": "Tab 5 Segmentation",
    "tab6": "Tab 6 Batch Correction",
    "tab7": "Tab 7 Phenotyping",
    "tab8": "Tab 8 Spatial Analysis",
    "tab9": "Tab 9 Visualization",
}

# channel -> (environment variable, shown when unset)
_CHANNELS = {
    "tab": ("SPACEMAP_TAB_LOGS", True),
    "perf": ("SPACEMAP_PERF_LOGS", True),
    "app": ("SPACEMAP_APP_LOGS", False),
}


def enabled(channel: str = "tab") -> bool:
    """Whether messages on ``channel`` should be printed."""
    master = os.getenv("SPACEMAP_CONSOLE_LOGS")
    if master is not None:
        return master == "1"
    variable, default = _CHANNELS.get(channel, _CHANNELS["tab"])
    value = os.getenv(variable)
    return default if value is None else value == "1"


# A timestamp some callers still put at the start of the message.
_LEADING_STAMP = re.compile(r"^\[?\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:[.,]\d+)?\]?\s*|^\[\d{2}:\d{2}:\d{2}\]\s*")


def format_line(tab: str, message: str, *, step: str | None = None, level: str = "info") -> str:
    source = TAB_LABELS.get(tab, tab)
    parts = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), source]
    if step:
        parts.append(step)
    text = _LEADING_STAMP.sub("", str(message).strip())
    if level in ("warning", "error") and not text.upper().startswith(level.upper()):
        text = f"{level.upper()}: {text}"
    parts.append(text)
    return " | ".join(parts)


def say(
    tab: str,
    message: str,
    *,
    step: str | None = None,
    level: str = "info",
    channel: str = "tab",
) -> None:
    """Print one progress line for ``tab`` (a key of ``TAB_LABELS``)."""
    if level in ("warning", "error") or enabled(channel):
        print(format_line(tab, message, step=step, level=level), flush=True)


@contextmanager
def timed(tab: str, label: str, *, step: str | None = None, channel: str = "tab"):
    """Report the start, the end and the duration of a block of work."""
    say(tab, f"Started: {label}", step=step, channel=channel)
    started = time.perf_counter()
    try:
        yield
    except Exception as exc:
        say(tab, f"Failed: {label} after {time.perf_counter() - started:.1f} s ({exc})",
            step=step, level="error", channel=channel)
        raise
    say(tab, f"Finished: {label} ({time.perf_counter() - started:.1f} s)", step=step, channel=channel)

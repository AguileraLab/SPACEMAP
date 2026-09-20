"""Small, conservative memory suggestions from user-written instructions."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MemorySuggestion:
    content: str
    kind: str
    confidence: float


_DURABLE_PATTERNS = (
    re.compile(r"\b(?:please\s+)?remember(?:\s+that)?\s+(.+)", re.IGNORECASE),
    re.compile(r"\bfrom now on[, :]?\s*(.+)", re.IGNORECASE),
)


def _clean_sentence(value: str) -> str:
    text = re.sub(r"\s+", " ", value).strip(" \t\r\n-:;,")
    if text and text[-1] not in ".!?":
        text += "."
    return text[:600]


def _kind_for(text: str) -> str:
    lowered = text.lower()
    if "marker" in lowered or "cluster" in lowered or "phenotype" in lowered:
        return "marker_alias"
    if "parameter" in lowered or "setting" in lowered or "recipe" in lowered:
        return "recipe_note"
    if any(word in lowered for word in ("prefer", "always", "never", "do not", "don't")):
        return "preference"
    return "lesson"


def suggest_memories(text: str, *, limit: int = 3) -> list[MemorySuggestion]:
    """Return only likely reusable user lessons; never treat them as verified."""
    clean = str(text or "").replace("\x00", " ").strip()
    if not clean:
        return []

    candidates: list[tuple[str, float]] = []
    for pattern in _DURABLE_PATTERNS:
        match = pattern.search(clean)
        if match:
            candidates.append((_clean_sentence(match.group(1)), 0.95))

    pieces = re.split(r"(?<=[.!?])\s+|\n+", clean)
    for piece in pieces:
        sentence = _clean_sentence(piece)
        lowered = sentence.lower()
        durable = any(
            phrase in lowered
            for phrase in (
                "i prefer ",
                "we prefer ",
                "i always ",
                "we always ",
                "i never ",
                "we never ",
                "i don't want ",
                "i do not want ",
                "must always ",
                "must never ",
            )
        )
        if durable:
            candidates.append((sentence, 0.85))

    suggestions: list[MemorySuggestion] = []
    seen: set[str] = set()
    for content, confidence in candidates:
        normalized = re.sub(r"\W+", " ", content.lower()).strip()
        if (
            len(content) < 12
            or content.endswith("?")
            or "/" in content
            or normalized in seen
        ):
            continue
        seen.add(normalized)
        suggestions.append(
            MemorySuggestion(
                content=content,
                kind=_kind_for(content),
                confidence=confidence,
            )
        )
        if len(suggestions) >= max(1, int(limit)):
            break
    return suggestions

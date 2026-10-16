"""Match project names without guessing between equally plausible choices."""

from difflib import SequenceMatcher
import re


class NameChoice(ValueError):
    def __init__(self, kind, value, choices):
        self.kind, self.value, self.choices = kind, value, list(choices)
        super().__init__(
            f'Choose which {kind} you mean by "{value}".' if choices else f'No matching {kind} was found for "{value}".'
        )


def normalise(value):
    value = re.sub(r"(?i)(\.ome)?\.(tiff?|qptiff|svs|csv|parquet)$", "", str(value).strip())
    return re.sub(r"[^\w]+", "", value.casefold()).replace("_", "")


def match(value, choices, kind, *, corrections=None):
    """Return the original name; report normalised/close matches and ambiguity."""
    choices = list(dict.fromkeys(str(c) for c in choices))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Choose a {kind}.")
    value = value.strip()
    if value in choices:
        return value
    key = normalise(value)
    exact = [c for c in choices if normalise(c) == key]
    if len(exact) > 1:
        raise NameChoice(kind, value, exact)
    if exact:
        answer = exact[0]
    else:
        ranked = sorted(((SequenceMatcher(None, key, normalise(c)).ratio(), c) for c in choices), reverse=True)
        close = [(score, name) for score, name in ranked if score >= 0.55][:6]
        if not close:
            raise NameChoice(kind, value, [])
        if close[0][0] < 0.90 or len(close) > 1 and close[0][0] - close[1][0] < 0.10:
            raise NameChoice(kind, value, [name for _, name in close])
        answer = close[0][1]
    if corrections is not None and answer != value:
        corrections.append({"kind": kind, "requested": value, "matched": answer})
    return answer

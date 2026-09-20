"""Text-rule parsing for the RESOLVE phenotyping table.

This module is intentionally independent of Streamlit so the rule language can
be tested without importing the full phenotyping UI.
"""

from __future__ import annotations

import re


_DIRECTIVE_RE = re.compile(r"(?i)\b(SUPPORT|NOT)\b\s*(?:=|:)?")
_MARKER_SPLIT_RE = re.compile(r"(?i)\bAND\b|\bOR\b|,")


def _unique(values: list[str]) -> list[str]:
    """Return values in first-seen order without duplicates."""
    return list(dict.fromkeys(values))


def _clean_marker_token(token: str) -> str:
    """Remove rule punctuation and optional matching quotes from a marker."""
    cleaned = token.strip().strip(" \t\r\n;(){}[]")
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _extract_bracket_groups(text: str) -> tuple[list[str], list[tuple[int, int]], str | None]:
    """Extract non-nested square-bracket groups, respecting quoted text."""
    groups: list[str] = []
    spans: list[tuple[int, int]] = []
    start: int | None = None
    quote: str | None = None
    escaped = False

    for position, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote is not None:
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            continue
        if quote is not None:
            continue
        if char == "[":
            if start is not None:
                return [], [], "nested '[' brackets are not allowed"
            start = position
        elif char == "]":
            if start is None:
                return [], [], "found ']' without a matching '['"
            groups.append(text[start + 1:position])
            spans.append((start, position + 1))
            start = None

    if quote is not None:
        return [], [], "a quoted marker is missing its closing quote"
    if start is not None:
        return [], [], "found '[' without a matching ']'"
    return groups, spans, None


def _outside_brackets(text: str, spans: list[tuple[int, int]]) -> str:
    """Replace bracketed portions with spaces while retaining outside syntax."""
    pieces: list[str] = []
    previous = 0
    for start, stop in spans:
        pieces.append(text[previous:start])
        pieces.append(" ")
        previous = stop
    pieces.append(text[previous:])
    return "".join(pieces)


def _split_group_tokens(content: str, *, allow_empty: bool) -> tuple[list[str], str | None]:
    """Split one marker group and detect accidental empty entries."""
    raw_tokens = _MARKER_SPLIT_RE.split(content)
    cleaned = [_clean_marker_token(token) for token in raw_tokens]
    nonempty = [token for token in cleaned if token]

    if not nonempty:
        if allow_empty:
            return [], None
        return [], "a must-have bracket group cannot be empty"
    if any(not token for token in cleaned):
        return [], "a marker list contains an empty entry"
    return _unique(nonempty), None


def _split_rule_sections(body: str) -> tuple[str, dict[str, list[str]], str | None]:
    """Split the required expression from optional SUPPORT and NOT sections."""
    matches = list(_DIRECTIVE_RE.finditer(body))
    if not matches:
        return body, {"SUPPORT": [], "NOT": []}, None

    sections: dict[str, list[str]] = {"SUPPORT": [], "NOT": []}
    primary = body[:matches[0].start()].strip(" \t\r\n,;()")
    for index, match in enumerate(matches):
        keyword = match.group(1).upper()
        if sections[keyword]:
            return primary, sections, f"{keyword} is specified more than once"
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        value = body[match.end():stop].strip(" \t\r\n,;()")
        # Store a sentinel for a present-but-empty directive. This keeps a
        # second directive detectable while still allowing SUPPORT=[] / NOT=[].
        sections[keyword].append(value)
    return primary, sections, None


def _parse_required_text(text: str) -> tuple[list[list[str]], str | None]:
    """Parse the bracket language or the original AND/OR language."""
    groups, spans, bracket_error = _extract_bracket_groups(text)
    if bracket_error:
        return [], bracket_error

    if groups or "[" in text or "]" in text:
        outside = _outside_brackets(text, spans)
        outside = re.sub(r"(?i)\bOR\b", "", outside)
        outside = outside.strip(" \t\r\n,;()")
        if outside:
            return [], f"unexpected text outside a bracket group: '{outside}'"

        alternatives: list[list[str]] = []
        for group in groups:
            tokens, token_error = _split_group_tokens(group, allow_empty=False)
            if token_error:
                return [], token_error
            alternatives.append(tokens)
        return alternatives, None

    alternatives = []
    for alternative in re.split(r"(?i)\bOR\b", text):
        tokens = [_clean_marker_token(token)
                  for token in re.split(r"(?i)\bAND\b|,", alternative)]
        tokens = _unique([token for token in tokens if token])
        if tokens:
            alternatives.append(tokens)
    return alternatives, None


def _parse_optional_sections(values: list[str]) -> tuple[list[str], str | None]:
    """Parse an optional SUPPORT or NOT list, bracketed or legacy-style."""
    if not values:
        return [], None
    text = values[0]
    groups, spans, bracket_error = _extract_bracket_groups(text)
    if bracket_error:
        return [], bracket_error

    tokens: list[str] = []
    if groups or "[" in text or "]" in text:
        outside = _outside_brackets(text, spans)
        outside = re.sub(r"(?i)\b(?:AND|OR)\b", "", outside)
        outside = outside.strip(" \t\r\n,;()")
        if outside:
            return [], f"unexpected text outside an optional marker list: '{outside}'"
        for group in groups:
            group_tokens, token_error = _split_group_tokens(group, allow_empty=True)
            if token_error:
                return [], token_error
            tokens.extend(group_tokens)
    else:
        tokens = [_clean_marker_token(token) for token in _MARKER_SPLIT_RE.split(text)]
    return _unique([token for token in tokens if token]), None


def parse_resolve_rule_text(text: str, valid_markers: list[str] | None = None) -> tuple[list[dict], list[str]]:
    """Parse one phenotype rule per line into RESOLVE table rows.

    Preferred grammar::

        CD8 = [CD8, CD3e, CD45] OR [CD8, CD3e]; SUPPORT=[]; NOT=[E-cadherin]

    Each bracket is an AND group and separate brackets are OR alternatives.
    Quotes around marker names are optional. The original unbracketed AND/OR
    syntax remains supported for saved rules and older projects.
    If ``valid_markers`` is None, validate syntax only and retain marker names
    for later resolution against a cluster expression table.
    """
    lookup = {str(marker).strip().lower(): str(marker).strip()
              for marker in (valid_markers or []) if str(marker).strip()}
    rows: list[dict] = []
    problems: list[str] = []
    seen_phenotypes: set[str] = set()

    for number, raw in enumerate(str(text or "").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        equals = line.find("=")
        colon = line.find(":")
        separators = [position for position in (equals, colon) if position >= 0]
        if not separators:
            problems.append(f"line {number}: expected 'Phenotype = markers'")
            continue
        split_at = min(separators)
        name = line[:split_at].strip()
        body = line[split_at + 1:].strip()
        if not name:
            problems.append(f"line {number}: no phenotype name")
            continue

        phenotype_key = name.lower()
        if phenotype_key in seen_phenotypes:
            problems.append(f"line {number}: duplicate phenotype '{name}' skipped")
            continue

        required_text, section_text, section_error = _split_rule_sections(body)
        if section_error:
            problems.append(f"line {number}: '{name}' skipped - {section_error}")
            continue
        alternatives, required_error = _parse_required_text(required_text)
        support, support_error = _parse_optional_sections(section_text["SUPPORT"])
        excluded, excluded_error = _parse_optional_sections(section_text["NOT"])
        syntax_error = required_error or support_error or excluded_error
        if syntax_error:
            problems.append(f"line {number}: '{name}' skipped - {syntax_error}")
            continue

        unknown: set[str] = set()

        def resolve(marker: str) -> str | None:
            if valid_markers is None:
                return marker
            canonical = lookup.get(marker.lower())
            if canonical is None:
                unknown.add(marker)
            return canonical

        resolved_alternatives = [
            [canonical for marker in alternative if (canonical := resolve(marker)) is not None]
            for alternative in alternatives
        ]
        resolved_support = [canonical for marker in support if (canonical := resolve(marker)) is not None]
        resolved_excluded = [canonical for marker in excluded if (canonical := resolve(marker)) is not None]

        if unknown:
            listed = ", ".join(sorted(unknown, key=str.lower))
            problems.append(f"line {number}: '{name}' skipped - unknown marker(s) {listed}")
            continue
        if not resolved_alternatives:
            problems.append(f"line {number}: '{name}' has no usable markers")
            continue

        seen_phenotypes.add(phenotype_key)
        rows.append({
            "Phenotype": name,
            "alternatives": resolved_alternatives,
            "support": resolved_support,
            "excluded": resolved_excluded,
        })

    return rows, problems

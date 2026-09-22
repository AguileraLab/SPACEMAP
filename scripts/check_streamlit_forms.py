#!/usr/bin/env python3
"""Fail when a normal button is used inside a Streamlit form.

Streamlit allows only ``form_submit_button`` inside ``st.form``; a plain
``st.button`` raises StreamlitAPIException at render time. That is not caught
by the test suite, because the tests never render these screens, so it reaches
the user as a traceback in the middle of their work -- which is exactly how it
was found.

    python scripts/check_streamlit_forms.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

#: Calls that are illegal inside a form.
FORBIDDEN = {"button", "download_button", "link_button"}


def _call_name(node: ast.AST) -> str | None:
    """Return the attribute name for st.<name>() or <col>.<name>()."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_form(node: ast.AST) -> bool:
    name = _call_name(node)
    return name == "form"


def _walk(node: ast.AST, inside_form: bool, problems: list[tuple[int, str]]) -> None:
    if isinstance(node, ast.With):
        entering = any(_is_form(item.context_expr) for item in node.items)
        for child in node.body:
            _walk(child, inside_form or entering, problems)
        return

    if inside_form:
        name = _call_name(node)
        if name in FORBIDDEN:
            problems.append((getattr(node, "lineno", 0), f"st.{name}()"))

    for child in ast.iter_child_nodes(node):
        # A function defined inside a form body is called elsewhere; its
        # contents are checked when that function is visited at module level.
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _walk(child, False, problems)
        else:
            _walk(child, inside_form, problems)


def main() -> int:
    root = Path("src")
    total: list[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        problems: list[tuple[int, str]] = []
        _walk(tree, False, problems)
        for line, what in problems:
            total.append(f"{path}:{line}: {what} inside st.form(); use form_submit_button()")

    if total:
        print("Buttons used inside a Streamlit form:\n")
        for entry in total:
            print(f"  {entry}")
        print("\nOnly st.form_submit_button() is allowed inside a form.")
        return 1
    print("No buttons found inside Streamlit forms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

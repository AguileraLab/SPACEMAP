"""Searchable, read-only index of the SPACEMAP scientific source tree."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .paths import (
    ASSISTANT_DATA_DIR,
    SOURCE_ROOT,
    is_within,
    require_assistant_write_path,
)


INDEX_PATH = ASSISTANT_DATA_DIR / "config" / "spacemap_source_index.json"
# SOURCE_ROOT is the directory holding the installed packages: 'src/' in a
# checkout, site-packages in an installation. Deriving it from the data root
# instead would index nothing once SPACEMAP is installed normally.
INDEXED_DIRECTORIES = (SOURCE_ROOT / "spacemap", SOURCE_ROOT / "spacemap_gui" / "tabs")
MAX_DOCSTRING = 1_200
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]{1,}")
QUERY_SYNONYMS = {
    "test": {"hpo", "optimize", "optimization", "grid", "search"},
    "tests": {"hpo", "optimize", "optimization", "grid", "search"},
    "parameter": {"hyperparameter", "hpo", "config"},
    "parameters": {"hyperparameter", "hpo", "config"},
    "quality": {"metric", "score", "validation", "silhouette"},
    "compare": {"comparison", "metric", "score"},
    "cluster": {"clustering", "leiden"},
    "clusters": {"clustering", "leiden"},
    "roi": {"region", "boundary", "geojson", "crop"},
    "register": {"registration", "align"},
    "background": {"subtract", "subtraction", "blank"},
    "blur": {"blurry", "focus", "artifact"},
    "blurry": {"blur", "focus", "artifact"},
    "segment": {"segmentation", "cellpose", "deepcell"},
    "segmentation": {"segment", "cellpose", "deepcell"},
    "phenotype": {"phenotyping", "cluster", "classification"},
    "phenotyping": {"phenotype", "cluster", "classification"},
    "spatial": {"neighborhood", "interaction", "density"},
}


@dataclass(frozen=True)
class SourceFunction:
    module: str
    qualname: str
    callable_name: str
    signature: str
    docstring: str
    path: str
    line: int
    end_line: int
    is_async: bool
    is_method: bool = False


def _source_files() -> list[Path]:
    files: list[Path] = []
    for directory in INDEXED_DIRECTORIES:
        if directory.is_dir():
            files.extend(path for path in directory.rglob("*.py") if "__pycache__" not in path.parts)
    return sorted(set(files))


def _fingerprint(files: list[Path]) -> dict[str, int]:
    count = 0
    total_size = 0
    newest_mtime_ns = 0
    for path in files:
        try:
            stat = path.stat()
        except OSError:
            continue
        count += 1
        total_size += int(stat.st_size)
        newest_mtime_ns = max(newest_mtime_ns, int(stat.st_mtime_ns))
    return {"file_count": count, "total_size": total_size, "newest_mtime_ns": newest_mtime_ns}


def _module_name(path: Path) -> str:
    relative = path.relative_to(SOURCE_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _argument_text(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    try:
        return ast.unparse(node.args)
    except Exception:
        return "..."


def _functions_in_file(path: Path) -> list[SourceFunction]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(text, filename=str(path))
    except (OSError, SyntaxError, ValueError):
        return []
    module = _module_name(path)
    records: list[SourceFunction] = []

    def visit(
        body: list[ast.stmt],
        parents: tuple[str, ...] = (),
        *,
        inside_class: bool = False,
    ) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = ".".join((*parents, node.name))
                records.append(
                    SourceFunction(
                        module=module,
                        qualname=qualname,
                        callable_name=f"{module}:{qualname}",
                        signature=f"{node.name}({_argument_text(node)})",
                        docstring=(ast.get_docstring(node) or "")[:MAX_DOCSTRING],
                        path=str(path.resolve()),
                        line=int(node.lineno),
                        end_line=int(getattr(node, "end_lineno", node.lineno)),
                        is_async=isinstance(node, ast.AsyncFunctionDef),
                        is_method=inside_class,
                    )
                )
                visit(node.body, (*parents, node.name), inside_class=inside_class)
            elif isinstance(node, ast.ClassDef):
                visit(node.body, (*parents, node.name), inside_class=True)

    visit(tree.body)
    return records


def rebuild_source_index() -> dict[str, Any]:
    """Rebuild metadata only; scientific source files remain untouched."""
    files = _source_files()
    functions = [record for path in files for record in _functions_in_file(path)]
    payload = {
        "schema_version": 2,
        "source_root": str(SOURCE_ROOT.resolve()),
        "fingerprint": _fingerprint(files),
        "functions": [asdict(record) for record in functions],
    }
    destination = require_assistant_write_path(INDEX_PATH)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return payload


def load_source_index(*, refresh_if_needed: bool = True) -> dict[str, Any]:
    files = _source_files()
    current = _fingerprint(files)
    try:
        payload = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return rebuild_source_index()
    if (
        refresh_if_needed
        and (payload.get("fingerprint") != current or payload.get("schema_version") != 2)
    ):
        return rebuild_source_index()
    return payload


def _tokens(value: str) -> set[str]:
    expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(value)).replace("_", " ")
    return {token.lower() for token in TOKEN_PATTERN.findall(expanded)}


def search_source(query: str, *, limit: int = 8) -> list[SourceFunction]:
    query_tokens = _tokens(query)
    query_tokens |= {
        synonym
        for token in tuple(query_tokens)
        for synonym in QUERY_SYNONYMS.get(token, set())
    }
    if not query_tokens:
        return []
    payload = load_source_index()
    scored: list[tuple[float, SourceFunction]] = []
    for item in payload.get("functions", []):
        try:
            record = SourceFunction(**item)
        except TypeError:
            continue
        name_tokens = _tokens(f"{record.module} {record.qualname} {record.signature}")
        doc_tokens = _tokens(record.docstring)
        exact_name = record.qualname.lower() in str(query).lower()
        score = len(query_tokens & name_tokens) * 4.0 + len(query_tokens & doc_tokens)
        if exact_name:
            score += 10.0
        if score > 0:
            scored.append((score, record))
    scored.sort(key=lambda pair: (-pair[0], pair[1].callable_name))
    return [record for _score, record in scored[: max(1, int(limit))]]


def is_planner_callable(record: SourceFunction) -> bool:
    """Match the public functions that the protected tool registry can expose."""
    if record.is_async:
        return False
    parts = record.qualname.split(".")
    if any(part.startswith("_") for part in parts):
        return False
    if record.is_method:
        # Public methods on backend classes are real scientific operations.
        # Nested functions and UI controller methods remain implementation details.
        return (
            record.module.startswith("spacemap.")
            and len(parts) == 2
        )
    if len(parts) != 1:
        return False
    if record.module.startswith("spacemap."):
        return True
    if not record.module.startswith("spacemap_gui.tabs."):
        return False
    return not record.qualname.casefold().startswith(
        ("render", "show_", "main", "run_app", "initialize_session")
    )


def read_function_source(record: SourceFunction, *, max_characters: int = 5_000) -> str:
    path = Path(record.path).resolve()
    if not is_within(path, SOURCE_ROOT) or not path.is_file():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    start = max(0, record.line - 1)
    end = min(len(lines), record.end_line)
    source = "\n".join(lines[start:end])
    if len(source) > max_characters:
        source = source[:max_characters] + "\n# ... source excerpt shortened ..."
    return source


def source_context_for_prompt(query: str, *, limit: int = 12, max_characters: int = 24_000) -> str:
    """Return source for the same registered callables shown in the tool catalog."""
    # Import at call time to avoid an import cycle: the registry builds its
    # catalog from this module's index records.
    from spacemap_workflows.registry import relevant_tools

    wanted = [item.callable_path for item in relevant_tools(query, limit=max(limit, 12))]
    all_records_by_name: dict[str, SourceFunction] = {}
    records_by_name: dict[str, SourceFunction] = {}
    for item in load_source_index().get("functions", []):
        try:
            record = SourceFunction(**item)
        except TypeError:
            continue
        all_records_by_name[record.callable_name] = record
        if is_planner_callable(record):
            records_by_name[record.callable_name] = record
    selected = [records_by_name[name] for name in wanted if name in records_by_name]
    selected_names = {item.callable_name for item in selected}
    if len(selected) < limit:
        for record in search_source(query, limit=limit * 3):
            if not is_planner_callable(record) or record.callable_name in selected_names:
                continue
            selected.append(record)
            selected_names.add(record.callable_name)
            if len(selected) >= limit:
                break

    records_to_include = selected[: max(1, int(limit))]
    per_record = max(900, max_characters // max(1, len(records_to_include)) - 100)
    chunks: list[str] = []
    used = 0
    for record in records_to_include:
        header = (
            f"CALLABLE: {record.callable_name}\n"
            f"SIGNATURE: {record.signature}\n"
            f"FILE: {record.path}:{record.line}\n"
        )
        body_limit = max(500, min(4_000, per_record - len(header) - 650))
        body = read_function_source(record, max_characters=body_limit)
        constructor_text = ""
        if record.is_method:
            class_name = record.qualname.split(".", 1)[0]
            constructor = all_records_by_name.get(f"{record.module}:{class_name}.__init__")
            if constructor is not None:
                constructor_text = (
                    f"\nCLASS CONSTRUCTOR: {constructor.signature}\n"
                    + read_function_source(constructor, max_characters=500)
                    + "\n"
                )
        chunk = header + constructor_text + "SOURCE EXCERPT:\n" + body
        if used + len(chunk) > max_characters:
            remaining = max_characters - used
            if remaining > 500:
                chunks.append(chunk[:remaining])
            break
        chunks.append(chunk)
        used += len(chunk)
    return "\n\n".join(chunks)

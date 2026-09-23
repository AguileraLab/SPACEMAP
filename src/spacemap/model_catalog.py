"""The catalog of offline models the assistant can install and run.

Earlier versions hard-coded a single model directory, a single environment and
a single port.  That made the assistant impossible to use with any other model
and impossible to update when a newer one appeared.  This module replaces that
with a small registry:

* a **built-in catalog** shipped with the package (``model_catalog.json``),
* a **user catalog** in the configuration directory that adds entries or
  overrides built-in ones by ``id``.

Nothing here downloads or loads a model; it only describes them.  That keeps
the catalog readable on a machine with no AI dependencies installed, which is
what the settings UI needs in order to offer a choice in the first place.

A catalog entry looks like this::

    {
      "id": "gemma-3-12b-it",
      "name": "Gemma 3 12B Instruct",
      "backend": "transformers",
      "repo_id": "google/gemma-3-12b-it",
      "local_dir": "google/gemma-3-12b-it",
      "approx_size_gb": 24,
      "min_vram_gb": 24,
      "cpu_capable": true,
      "gated": true
    }

``backend`` is ``transformers`` for a Hugging Face model directory or ``gguf``
for a single quantized file run through ``llama-cpp-python``.  ``gated`` marks
a model whose license must be accepted on Hugging Face before download, which
the UI surfaces so the user is not left staring at a 401.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

from .ai_settings import resolved_models_dir
from .userpaths import config_dir

BUILTIN_CATALOG_FILE = Path(__file__).resolve().parent / "model_catalog.json"
USER_CATALOG_FILENAME = "model_catalog.json"

BACKENDS = ("transformers", "gguf")


class ModelCatalogError(RuntimeError):
    """Raised when a catalog entry is malformed or unknown."""


@dataclass(frozen=True)
class ModelEntry:
    """One installable model."""

    id: str
    name: str
    backend: str = "transformers"
    publisher: str = ""
    repo_id: str = ""
    revision: str = "main"
    local_dir: str = ""
    filename: str = ""
    allow_patterns: tuple[str, ...] = ()
    approx_size_gb: float = 0.0
    min_vram_gb: float = 0.0
    cpu_capable: bool = True
    gated: bool = False
    modalities: tuple[str, ...] = ("text",)
    license: str = ""
    license_url: str = ""
    notes: str = ""
    source: str = "builtin"

    def target_dir(self) -> Path:
        """The directory this model occupies inside the models folder."""
        relative = self.local_dir or self.repo_id or self.id
        return resolved_models_dir() / relative

    def target_file(self) -> Path | None:
        """For ``gguf`` entries, the single weights file on disk."""
        if self.backend != "gguf" or not self.filename:
            return None
        return self.target_dir() / self.filename

    def is_installed(self) -> bool:
        """True when the weights appear to be present locally."""
        if self.backend == "gguf":
            target = self.target_file()
            return bool(target and target.is_file() and target.stat().st_size > 0)
        directory = self.target_dir()
        if not directory.is_dir():
            return False
        # A Hugging Face model directory is usable once it has a config and at
        # least one weight shard; an interrupted download leaves the config
        # behind on its own, so both checks are needed.
        if not (directory / "config.json").is_file():
            return False
        return any(directory.glob("*.safetensors")) or any(directory.glob("*.bin"))

    def installed_size_bytes(self) -> int:
        """Bytes currently on disk for this model (0 when absent)."""
        directory = self.target_dir()
        if not directory.is_dir():
            return 0
        total = 0
        for path in directory.rglob("*"):
            if path.is_file() and not path.is_symlink():
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
        return total

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["allow_patterns"] = list(self.allow_patterns)
        payload["modalities"] = list(self.modalities)
        return payload


def _coerce(raw: dict[str, Any], *, source: str) -> ModelEntry:
    """Build a :class:`ModelEntry` from catalog JSON, rejecting bad records."""
    identifier = str(raw.get("id") or "").strip()
    if not identifier:
        raise ModelCatalogError("A model entry needs an 'id'.")
    backend = str(raw.get("backend") or "transformers").strip()
    if backend not in BACKENDS:
        raise ModelCatalogError(
            f"Model '{identifier}' uses unknown backend '{backend}'; expected one of {BACKENDS}."
        )
    return ModelEntry(
        id=identifier,
        name=str(raw.get("name") or identifier),
        backend=backend,
        publisher=str(raw.get("publisher") or ""),
        repo_id=str(raw.get("repo_id") or ""),
        revision=str(raw.get("revision") or "main"),
        local_dir=str(raw.get("local_dir") or ""),
        filename=str(raw.get("filename") or ""),
        allow_patterns=tuple(str(item) for item in raw.get("allow_patterns") or ()),
        approx_size_gb=float(raw.get("approx_size_gb") or 0.0),
        min_vram_gb=float(raw.get("min_vram_gb") or 0.0),
        cpu_capable=bool(raw.get("cpu_capable", True)),
        gated=bool(raw.get("gated", False)),
        modalities=tuple(str(item) for item in raw.get("modalities") or ("text",)),
        license=str(raw.get("license") or ""),
        license_url=str(raw.get("license_url") or ""),
        notes=str(raw.get("notes") or ""),
        source=source,
    )


def _read_catalog_file(path: Path, *, source: str) -> list[ModelEntry]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    records = payload.get("models") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return []
    entries = []
    for record in records:
        if not isinstance(record, dict):
            continue
        try:
            entries.append(_coerce(record, source=source))
        except ModelCatalogError:
            # One bad hand-edited entry must not hide the rest of the catalog.
            continue
    return entries


def user_catalog_path() -> Path:
    """Location of the user-editable catalog."""
    return config_dir() / USER_CATALOG_FILENAME


def builtin_models() -> list[ModelEntry]:
    """Entries shipped with the package."""
    return _read_catalog_file(BUILTIN_CATALOG_FILE, source="builtin")


def user_models() -> list[ModelEntry]:
    """Entries the user added through the UI or by editing the catalog."""
    return _read_catalog_file(user_catalog_path(), source="user")


def list_models() -> list[ModelEntry]:
    """Every known model, with user entries overriding built-ins by id."""
    merged: dict[str, ModelEntry] = {entry.id: entry for entry in builtin_models()}
    for entry in user_models():
        merged[entry.id] = entry
    return sorted(merged.values(), key=lambda item: (not item.is_installed(), item.name.lower()))


def get_model(model_id: str) -> ModelEntry:
    """Return one model by id."""
    for entry in list_models():
        if entry.id == model_id:
            return entry
    raise ModelCatalogError(f"No model named '{model_id}' is in the catalog.")


def installed_models() -> list[ModelEntry]:
    """Models whose weights are present on this machine."""
    return [entry for entry in list_models() if entry.is_installed()]


def save_user_models(entries: Iterable[ModelEntry | dict[str, Any]]) -> Path:
    """Replace the user catalog with *entries*."""
    records = []
    for entry in entries:
        record = entry.to_dict() if isinstance(entry, ModelEntry) else dict(entry)
        record.pop("source", None)
        records.append(record)
    target = user_catalog_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps({"schema_version": 1, "models": records}, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def add_user_model(record: dict[str, Any]) -> ModelEntry:
    """Add or replace a user catalog entry and return the stored model."""
    entry = _coerce(record, source="user")
    existing = [item for item in user_models() if item.id != entry.id]
    save_user_models([*existing, entry])
    return entry


def remove_user_model(model_id: str) -> bool:
    """Remove a user catalog entry.  Built-in entries cannot be removed."""
    existing = user_models()
    remaining = [item for item in existing if item.id != model_id]
    if len(remaining) == len(existing):
        return False
    save_user_models(remaining)
    return True


__all__ = [
    "BACKENDS",
    "ModelCatalogError",
    "ModelEntry",
    "add_user_model",
    "builtin_models",
    "get_model",
    "installed_models",
    "list_models",
    "remove_user_model",
    "save_user_models",
    "user_catalog_path",
    "user_models",
]

"""Find open models on Hugging Face and turn a result into a catalog entry.

The AI setup page searches here, then adds the chosen result to the user's
model catalog (``spacemap.model_catalog.add_user_model``); from there it is
downloaded, used, started and stopped like the built-in models. Only models
with ``safetensors`` weights are offered: those are the ones SPACEMAP's model
server loads (text-only or text-and-image chat models).
"""

from __future__ import annotations

import re
from typing import Any

#: Bytes per parameter for the dtypes Hugging Face reports.
_BYTES = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I64": 8, "I32": 4,
          "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


class ModelSearchError(RuntimeError):
    """The search could not be run (no network, no huggingface_hub, bad reply)."""


def _size_gb(safetensors: Any) -> tuple[float, float]:
    """(approximate download size in GB, parameters in billions) from the safetensors summary."""
    parameters = getattr(safetensors, "parameters", None) or {}
    total = getattr(safetensors, "total", None) or sum(parameters.values())
    size = sum(count * _BYTES.get(str(dtype).upper(), 2) for dtype, count in parameters.items())
    return round(size / 1e9, 1), round((total or 0) / 1e9, 1)


def _license(info: Any) -> str:
    card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
    value = card.get("license") if hasattr(card, "get") else getattr(card, "license", None)
    if value:
        return str(value)
    for tag in getattr(info, "tags", None) or []:
        if str(tag).startswith("license:"):
            return str(tag).split(":", 1)[1]
    return ""


def search_models(query: str, *, images: bool = False, limit: int = 12, api: Any = None) -> list[dict]:
    """Chat models on Hugging Face matching *query*, most downloaded first."""
    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ModelSearchError("Searching needs the assistant's packages (huggingface_hub).") from exc
        api = HfApi()
    try:
        found = list(api.list_models(
            search=query.strip() or None,
            pipeline_tag="image-text-to-text" if images else "text-generation",
            sort="downloads", limit=int(limit) * 3,
            expand=["downloads", "likes", "gated", "safetensors", "cardData", "library_name", "tags"],
        ))
    except Exception as exc:  # network errors come in many types
        raise ModelSearchError(f"Hugging Face could not be reached: {exc}") from exc
    results = []
    for info in found:
        safetensors = getattr(info, "safetensors", None)
        if not safetensors:
            continue  # GGUF-only or other formats the model server does not load
        size_gb, billions = _size_gb(safetensors)
        repo = str(info.id)
        results.append({
            "repo_id": repo,
            "name": repo.split("/", 1)[-1],
            "publisher": repo.split("/", 1)[0] if "/" in repo else "",
            "downloads": int(getattr(info, "downloads", 0) or 0),
            "likes": int(getattr(info, "likes", 0) or 0),
            "size_gb": size_gb,
            "parameters_b": billions,
            "gated": bool(getattr(info, "gated", False)),
            "license": _license(info),
            "images": bool(images),
        })
        if len(results) >= limit:
            break
    return results


def catalog_record(result: dict) -> dict:
    """The user-catalog entry for a search result."""
    repo = result["repo_id"]
    identifier = re.sub(r"[^a-z0-9.]+", "-", repo.lower()).strip("-")
    size = float(result.get("size_gb") or 0.0)
    return {
        "id": identifier,
        "name": result.get("name") or repo,
        "backend": "transformers",
        "publisher": result.get("publisher", ""),
        "repo_id": repo,
        "local_dir": repo,
        "approx_size_gb": size,
        "min_vram_gb": round(size * 1.15, 1),  # full quality; 4-bit needs about a third of this
        "cpu_capable": size <= 10,
        "gated": bool(result.get("gated")),
        "modalities": ["text", "image"] if result.get("images") else ["text"],
        "license": result.get("license", ""),
        "license_url": f"https://huggingface.co/{repo}" if result.get("gated") else "",
        "notes": f"Added from a Hugging Face search ({int(result.get('downloads') or 0):,} downloads).",
    }

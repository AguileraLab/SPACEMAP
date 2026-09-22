"""Fast, durable storage for labels assigned in the interface.

Saving one phenotype used to rewrite the whole unified table: measured at about
13 minutes for a 13 GB CSV, plus another 2.7 minutes to rebuild the Parquet
copy the rewrite invalidated. Labelling fifty cells in sittings was impractical.

Labels are now appended to a small sidecar beside the table and merged over it
whenever it is read, so a save takes well under a second and the interface
always shows current values. The sidecar holds one row per assignment:

    row      position of the cell in the unified table
    column   the column being written, e.g. phenotype_manual
    value    the label
    saved_at when it was assigned

Folding the sidecar into the CSV and Parquet is a separate, deliberate step --
:func:`commit`. Until then the CSV on disk does not contain the newest labels,
which is the trade the speed buys.

**Row positions are the key.** They are stable as long as the unified table is
not reordered or re-exported behind SPACEMAP's back. :func:`load` records the
table's size and modification time when the first label is written and refuses
to apply labels to a table that no longer matches, rather than writing a label
onto the wrong cell.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SIDECAR_SUFFIX = ".labels.parquet"
_FINGERPRINT_KEY = b"spacemap_table"


class LabelStoreError(RuntimeError):
    """Raised when labels cannot be stored or applied safely."""


def sidecar_path(table_path: str | Path) -> Path:
    """Where the pending labels for *table_path* are kept."""
    source = Path(table_path).expanduser()
    beside = source.with_suffix(source.suffix + SIDECAR_SUFFIX)
    if os.access(source.parent, os.W_OK):
        return beside
    from hashlib import sha1

    digest = sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:16]
    root = Path(os.environ.get("SPACEMAP_CACHE_DIR", Path.home() / ".cache" / "spacemap")).expanduser()
    return root / "labels" / f"{source.stem}.{digest}{SIDECAR_SUFFIX}"


def _fingerprint(table_path: Path) -> str:
    stat = table_path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def load(table_path: str | Path) -> pd.DataFrame:
    """Return the pending labels for a table, newest assignment per cell.

    Returns an empty frame when there are none, or when the table has changed
    since the labels were written: applying positions to a reordered table
    would silently label the wrong cells.
    """
    source = Path(table_path).expanduser()
    target = sidecar_path(source)
    empty = pd.DataFrame(columns=["row", "column", "value", "saved_at"])
    if not target.is_file() or not source.is_file():
        return empty

    try:
        import pyarrow.parquet as pq

        schema = pq.read_schema(target)
        recorded = (schema.metadata or {}).get(_FINGERPRINT_KEY)
        if recorded is not None and recorded.decode("utf-8") != _fingerprint(source):
            logger.warning(
                "Pending labels for %s were written against a different version of the "
                "table and will not be applied. Commit or remove %s.",
                source.name,
                target.name,
            )
            return empty
        frame = pd.read_parquet(target)
    except Exception as exc:
        logger.warning("Could not read pending labels from %s: %s", target, exc)
        return empty

    if frame.empty:
        return empty
    # The sidecar is append-only, so the last write for a cell and column wins.
    frame = frame.sort_values("saved_at").drop_duplicates(subset=["row", "column"], keep="last")
    return frame.reset_index(drop=True)


def append(table_path: str | Path, updates: Iterable[dict[str, Any]]) -> int:
    """Record label assignments.  Returns how many were stored.

    Each update needs ``row`` (position in the unified table), ``column`` and
    ``value``.  Appending is what makes a save fast: nothing else is rewritten.
    """
    source = Path(table_path).expanduser()
    if not source.is_file():
        raise LabelStoreError(f"{source} does not exist.")

    records = []
    stamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    for update in updates:
        if "row" not in update or "column" not in update:
            continue
        records.append(
            {
                "row": int(update["row"]),
                "column": str(update["column"]),
                "value": "" if update.get("value") is None else str(update["value"]),
                "saved_at": str(update.get("saved_at") or stamp),
            }
        )
    if not records:
        return 0

    existing = load(source)
    combined = pd.concat([existing, pd.DataFrame(records)], ignore_index=True)
    combined = combined.sort_values("saved_at").drop_duplicates(subset=["row", "column"], keep="last")

    target = sidecar_path(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(combined.reset_index(drop=True), preserve_index=False)
        table = table.replace_schema_metadata({_FINGERPRINT_KEY: _fingerprint(source).encode("utf-8")})
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        pq.write_table(table, temporary, compression="snappy")
        os.replace(temporary, target)
    except Exception as exc:
        raise LabelStoreError(f"Could not save labels: {exc}") from exc
    return len(records)


def pending_count(table_path: str | Path) -> int:
    """How many assignments are waiting to be folded into the table."""
    return int(len(load(table_path)))


def pending_columns(table_path: str | Path) -> list[str]:
    """Which columns have pending assignments."""
    frame = load(table_path)
    return sorted(frame["column"].unique().tolist()) if not frame.empty else []


def apply_to(
    frame: pd.DataFrame,
    table_path: str | Path,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Overlay pending labels onto a frame loaded from *table_path*.

    *frame* must carry the table's own row order, which is what the loader
    returns. Only label columns present in *frame* -- or named in *columns* --
    are written, so a narrow column read stays narrow.
    """
    labels = load(table_path)
    if labels.empty:
        return frame

    wanted = set(columns) if columns else None
    touched = False
    for column, group in labels.groupby("column"):
        name = str(column)
        if wanted is not None and name not in wanted:
            continue
        if name not in frame.columns:
            if wanted is None:
                continue
            frame[name] = pd.Series([pd.NA] * len(frame), index=frame.index, dtype=object)
        positions = group["row"].to_numpy()
        inside = (positions >= 0) & (positions < len(frame))
        if not inside.any():
            continue
        # Assigning text into a numeric column needs an object column first,
        # or pandas raises rather than widening.
        if frame[name].dtype != object:
            frame[name] = frame[name].astype(object)
        values = group["value"].to_numpy(dtype=object)[inside]
        # An empty value means the label was cleared, which must read back as a
        # missing value rather than an empty string, or a cleared cell still
        # counts as assigned.
        values = np.where(
            np.asarray([str(item).strip() == "" for item in values]), np.nan, values
        )
        frame.iloc[positions[inside], frame.columns.get_loc(name)] = values
        touched = True
    if touched:
        logger.info("Applied %d pending label(s) from %s", len(labels), sidecar_path(table_path).name)
    return frame


def clear(table_path: str | Path) -> bool:
    """Discard pending labels.  Returns False when there were none."""
    target = sidecar_path(table_path)
    if not target.is_file():
        return False
    target.unlink()
    return True


def commit(
    table_path: str | Path,
    *,
    progress: Any | None = None,
    rebuild_fast_copy: bool = True,
) -> dict[str, Any]:
    """Fold pending labels into the CSV, and refresh the Parquet copy.

    This is the slow operation the sidecar exists to avoid doing on every save:
    it rewrites the whole table. Call it when the labelling session is done.
    """
    from . import dataio

    source = Path(table_path).expanduser()
    labels = load(source)
    report: dict[str, Any] = {"applied": int(len(labels)), "columns": [], "written": False}
    if labels.empty:
        return report

    def say(message: str) -> None:
        logger.info(message)
        if progress is not None:
            try:
                progress(message)
            except Exception:
                pass

    say(f"Reading {source.name}...")
    frame = dataio.load(source, apply_labels=False)

    say(f"Applying {len(labels)} label(s)...")
    frame = apply_to(frame, source, columns=sorted(labels["column"].unique().tolist()))
    report["columns"] = sorted(labels["column"].unique().tolist())

    say("Writing the CSV; this is the slow part...")
    temporary = source.with_name(f".{source.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, source)
    except Exception as exc:
        Path(temporary).unlink(missing_ok=True)
        raise LabelStoreError(f"Could not write {source}: {exc}") from exc
    report["written"] = True

    # The labels are in the table now, so the sidecar must go before anything
    # reads it against the new fingerprint.
    clear(source)

    if rebuild_fast_copy:
        say("Rebuilding the fast copy...")
        dataio.ensure_fast_copy(source, force=True, progress=progress)
    say("Done.")
    return report


__all__ = [
    "LabelStoreError",
    "append",
    "apply_to",
    "clear",
    "commit",
    "load",
    "pending_columns",
    "pending_count",
    "sidecar_path",
]

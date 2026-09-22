"""Column-aware loading for the large per-cell tables.

The batch-corrected tables reach 13 GB and 133 columns, while a given screen
usually needs two to five of those columns.  Reading the whole file to answer a
two-column question costs about four minutes and fifteen gigabytes of memory,
and it happens again on every Streamlit rerun.

This module is the single place the GUI asks for data, so that the cost is paid
once and only for the columns actually wanted:

    columns(path)                     -> the column names, reading no data
    load(path, columns=[...])         -> only those columns
    ensure_fast_copy(path)            -> build the Parquet twin

**Why a Parquet twin is needed.**  ``pandas.read_csv(usecols=...)`` saves memory
but barely saves time: a CSV is one run of text, so reaching column 68 of row
5,000,000 means parsing every character before it.  Measured on the 8.6M-row
PDAC table, a two-column read takes 72 s from CSV and 1.1 s from Parquet.
Parquet stores each column separately, so the columns not asked for are never
read from disk.

The twin sits beside the CSV, is regenerated whenever the CSV changes, and is
safe to delete at any time.  The CSV remains the file users import, export and
open in a spreadsheet; nothing in the project's own layout changes.

Nothing here imports Streamlit, so it can be tested and reused from scripts and
notebooks.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

#: Files smaller than this are fast enough to read directly; building a twin
#: for them would cost more than it saves.
FAST_COPY_MIN_BYTES = 200 * 1024 * 1024

#: Rows sampled to infer column types before converting.  Streaming conversion
#: infers types per block, and several columns in real tables are blank for the
#: first few million rows, so a block-local guess of "empty" later collides with
#: a numeric value and aborts the conversion.
SCHEMA_SAMPLE_ROWS = 2_000_000

#: Columns stored as ``category`` when present.  Measured 62x smaller than the
#: default text dtype on the PDAC table, and it speeds up ``groupby``.
DEFAULT_CATEGORICAL_HINTS = (
    "batch_id",
    "slide_id",
    "sample",
    "sample_id",
    "patient",
    "patient_id",
    "roi",
    "tissue",
    "region",
    "compartment",
    "condition",
    "group",
)

#: Any column whose name starts with one of these is also treated as
#: categorical; this covers phenotype_leiden, phenotype_som and so on.
CATEGORICAL_PREFIXES = ("phenotype", "cell_type", "celltype", "cluster", "annotation")

_METADATA_KEY = b"spacemap_source"


#: Builds currently running, keyed by resolved source path.  A build takes
#: minutes on a large table, so it must never run twice for the same file and
#: must never block the interface.
_BUILDS: dict[str, threading.Thread] = {}
_BUILD_LOCK = threading.Lock()
_BUILD_ERRORS: dict[str, str] = {}


class DataIOError(RuntimeError):
    """Raised when a table cannot be read as asked."""


def fast_copies_enabled() -> bool:
    """False when an administrator or user has switched Parquet twins off."""
    return os.environ.get("SPACEMAP_FAST_COPY", "").strip().lower() not in {
        "0",
        "off",
        "false",
        "no",
        "disable",
        "disabled",
    }


def _source_fingerprint(path: Path) -> str:
    """Identify a source file by size and modification time."""
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def fast_copy_path(path: str | Path) -> Path:
    """Where the Parquet twin for *path* lives.

    Beside the CSV when that directory is writable, so the twin travels with
    the project and is obvious to anyone looking at the folder.  Otherwise in
    the user cache, keyed by the full source path, for read-only shares.
    """
    source = Path(path).expanduser().resolve()
    beside = source.with_suffix(source.suffix + ".parquet")
    if os.access(source.parent, os.W_OK):
        return beside

    from hashlib import sha1

    digest = sha1(str(source).encode("utf-8")).hexdigest()[:16]
    return _cache_root() / "fast_copies" / f"{source.stem}.{digest}.parquet"


def _cache_root() -> Path:
    """User-writable cache location, used when the data folder is read-only.

    Prefers the shared location when this build provides one, so a single
    setting moves every cache at once; otherwise falls back to the usual
    per-user cache directory.
    """
    try:
        from .userpaths import cache_dir

        return cache_dir()
    except Exception:
        configured = os.environ.get("SPACEMAP_CACHE_DIR", "").strip()
        if configured:
            return Path(configured).expanduser()
        return Path.home() / ".cache" / "spacemap"


def _twin_matches_source(twin: Path, source: Path) -> bool:
    """True when *twin* was built from the current contents of *source*."""
    try:
        import pyarrow.parquet as pq

        metadata = pq.read_schema(twin).metadata or {}
    except Exception:
        return False
    recorded = metadata.get(_METADATA_KEY)
    if recorded is None:
        return False
    try:
        return recorded.decode("utf-8") == _source_fingerprint(source)
    except (UnicodeDecodeError, OSError):
        return False


def fast_copy_info(path: str | Path) -> dict[str, Any]:
    """Describe the twin for *path*, for display in the interface."""
    source = Path(path).expanduser()
    info: dict[str, Any] = {
        "source": str(source),
        "source_bytes": 0,
        "twin": str(fast_copy_path(source)) if source.exists() else "",
        "twin_bytes": 0,
        "exists": False,
        "current": False,
        "worth_building": False,
        "enabled": fast_copies_enabled(),
    }
    if not source.is_file():
        return info
    info["source_bytes"] = source.stat().st_size
    info["worth_building"] = info["source_bytes"] >= FAST_COPY_MIN_BYTES
    twin = fast_copy_path(source)
    if twin.is_file():
        info["exists"] = True
        info["twin_bytes"] = twin.stat().st_size
        info["current"] = _twin_matches_source(twin, source)
    return info


def _arrow_schema_from_sample(source: Path) -> dict[str, Any]:
    """Infer a column type map from the first rows of the CSV.

    Columns that are entirely blank in the sample are typed as float rather
    than null, because a null-typed column aborts the conversion as soon as a
    real value appears later in the file.
    """
    import numpy as np
    import pyarrow as pa

    sample = pd.read_csv(source, nrows=SCHEMA_SAMPLE_ROWS, low_memory=False)
    types: dict[str, Any] = {}
    for name in sample.columns:
        dtype = sample[name].dtype
        if sample[name].isna().all() or np.issubdtype(dtype, np.floating):
            types[name] = pa.float64()
        elif np.issubdtype(dtype, np.integer):
            types[name] = pa.int64()
        elif np.issubdtype(dtype, np.bool_):
            types[name] = pa.bool_()
        else:
            types[name] = pa.string()
    return types


def ensure_fast_copy(
    path: str | Path,
    *,
    force: bool = False,
    progress: Any | None = None,
) -> Path | None:
    """Build or refresh the Parquet twin for *path*; return it, or None.

    Returns None when twins are disabled, the file is small enough not to
    need one, or the conversion fails.  A failure is logged and never raised:
    the caller falls back to reading the CSV.

    *progress* may be any callable taking a status string, so a caller can
    surface what is happening without this module importing Streamlit.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise DataIOError(f"{source} does not exist.")
    if not fast_copies_enabled():
        return None

    twin = fast_copy_path(source)
    if twin.is_file() and not force and _twin_matches_source(twin, source):
        return twin
    if source.stat().st_size < FAST_COPY_MIN_BYTES and not force:
        return None

    def report(message: str) -> None:
        logger.info(message)
        if progress is not None:
            try:
                progress(message)
            except Exception:
                pass

    try:
        import pyarrow as pa
        import pyarrow.csv as pv
        import pyarrow.parquet as pq
    except ImportError as exc:
        logger.warning("pyarrow is unavailable, so no fast copy was built: %s", exc)
        return None

    try:
        report("Reading column types...")
        column_types = _arrow_schema_from_sample(source)

        report(f"Converting {source.name} to a fast-read copy (once)...")
        twin.parent.mkdir(parents=True, exist_ok=True)
        # Written to a temporary name and renamed, so a crash or a second
        # session cannot leave a half-written twin that looks usable.
        temporary = twin.with_name(f".{twin.name}.{os.getpid()}.tmp")
        fingerprint = _source_fingerprint(source)

        reader = pv.open_csv(
            source,
            read_options=pv.ReadOptions(use_threads=True, block_size=256 << 20),
            convert_options=pv.ConvertOptions(
                strings_can_be_null=True, column_types=column_types
            ),
        )
        writer = None
        rows = 0
        try:
            for batch in reader:
                table = pa.Table.from_batches([batch])
                if writer is None:
                    schema = table.schema.with_metadata({_METADATA_KEY: fingerprint})
                    writer = pq.ParquetWriter(temporary, schema, compression="snappy")
                writer.write_table(table.replace_schema_metadata({_METADATA_KEY: fingerprint}))
                rows += table.num_rows
                if rows % 2_000_000 < table.num_rows:
                    report(f"Converted {rows:,} rows...")
        finally:
            if writer is not None:
                writer.close()
            reader.close()

        os.replace(temporary, twin)
        report(f"Fast copy ready: {rows:,} rows.")
        return twin
    except Exception as exc:
        logger.warning("Could not build a fast copy of %s: %s", source, exc)
        for leftover in twin.parent.glob(f".{twin.name}.*.tmp"):
            try:
                leftover.unlink()
            except OSError:
                pass
        return None


def is_building(path: str | Path) -> bool:
    """True when a fast copy for *path* is being built right now."""
    key = str(Path(path).expanduser().resolve())
    with _BUILD_LOCK:
        thread = _BUILDS.get(key)
        return bool(thread and thread.is_alive())


def build_error(path: str | Path) -> str:
    """The reason the last background build failed, or an empty string."""
    key = str(Path(path).expanduser().resolve())
    with _BUILD_LOCK:
        return _BUILD_ERRORS.get(key, "")


def start_fast_copy(path: str | Path) -> bool:
    """Begin building the fast copy in the background; return True if started.

    Returns False when one is already current, already being built, disabled,
    or the file is small enough not to need one.  The caller carries on reading
    the CSV in the meantime, so nothing waits for this.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file() or not fast_copies_enabled():
        return False
    if source.stat().st_size < FAST_COPY_MIN_BYTES:
        return False
    twin = fast_copy_path(source)
    if twin.is_file() and _twin_matches_source(twin, source):
        return False

    key = str(source)
    with _BUILD_LOCK:
        existing = _BUILDS.get(key)
        if existing and existing.is_alive():
            return False
        _BUILD_ERRORS.pop(key, None)

        def worker() -> None:
            try:
                result = ensure_fast_copy(source, force=True)
                if result is None:
                    with _BUILD_LOCK:
                        _BUILD_ERRORS[key] = "The fast copy could not be built."
            except Exception as exc:  # noqa: BLE001 - reported, never raised
                with _BUILD_LOCK:
                    _BUILD_ERRORS[key] = str(exc)

        thread = threading.Thread(
            target=worker, name=f"spacemap-fastcopy-{source.name}", daemon=True
        )
        _BUILDS[key] = thread
    thread.start()
    return True


def clear_fast_copy(path: str | Path) -> bool:
    """Delete the Parquet twin for *path*.  Returns False when none existed."""
    twin = fast_copy_path(path)
    if not twin.is_file():
        return False
    twin.unlink()
    return True


def verify_fast_copy(
    path: str | Path,
    *,
    sample_rows: int = 200_000,
    full: bool = False,
) -> dict[str, Any]:
    """Check that the Parquet twin still agrees with its CSV.

    The twin is derived data and is rebuilt whenever the CSV changes, so it
    should never drift.  This exists so that can be demonstrated rather than
    assumed: it compares the row count, the column names and order, and the
    values themselves.

    By default the value comparison uses the first *sample_rows* rows, which is
    quick; ``full=True`` compares every row, which on a 13 GB table takes a few
    minutes.  Returns a report with ``ok`` set when everything matched.
    """
    source = Path(path).expanduser().resolve()
    report: dict[str, Any] = {
        "ok": False,
        "checked": [],
        "problems": [],
        "rows_compared": 0,
        "source_rows": None,
        "twin_rows": None,
    }
    twin = fast_copy_path(source)
    if not twin.is_file():
        report["problems"].append("No fast copy exists for this file.")
        return report
    if not _twin_matches_source(twin, source):
        report["problems"].append(
            "The fast copy was built from a different version of the CSV. "
            "Rebuild it before comparing."
        )
        return report

    try:
        import pyarrow.parquet as pq

        twin_meta = pq.read_metadata(twin)
        report["twin_rows"] = twin_meta.num_rows
        twin_columns = list(pq.read_schema(twin).names)

        csv_columns = list(pd.read_csv(source, nrows=0).columns)
        if twin_columns != csv_columns:
            only_csv = [c for c in csv_columns if c not in twin_columns]
            only_twin = [c for c in twin_columns if c not in csv_columns]
            if only_csv or only_twin:
                report["problems"].append(
                    f"Column mismatch. Only in CSV: {only_csv[:5]}; "
                    f"only in fast copy: {only_twin[:5]}."
                )
            else:
                report["problems"].append("Columns are in a different order.")
        else:
            report["checked"].append(f"{len(csv_columns)} columns match, in order")

        limit = None if full else sample_rows
        csv_frame = _read_csv_head(source, limit)
        twin_frame = pd.read_parquet(twin)
        if limit is not None:
            twin_frame = twin_frame.head(limit)
        report["rows_compared"] = len(csv_frame)

        if full:
            report["source_rows"] = len(csv_frame)
            if report["source_rows"] != report["twin_rows"]:
                report["problems"].append(
                    f"Row count differs: CSV {report['source_rows']:,}, "
                    f"fast copy {report['twin_rows']:,}."
                )
            else:
                report["checked"].append(f"{report['twin_rows']:,} rows in both")

        shared = [c for c in csv_columns if c in twin_columns]
        mismatched = []
        for name in shared:
            left = csv_frame[name]
            right = twin_frame[name].reset_index(drop=True)
            if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
                # Both sides came from the same decimal text, so any difference
                # would be a conversion fault rather than rounding drift.
                same = bool(
                    len(left) == len(right)
                    and ((left.values == right.values) | (left.isna().values & right.isna().values)).all()
                )
            else:
                same = bool(
                    len(left) == len(right)
                    and (
                        (left.astype("object").values == right.astype("object").values)
                        | (left.isna().values & right.isna().values)
                    ).all()
                )
            if not same:
                mismatched.append(name)
        if mismatched:
            report["problems"].append(
                f"{len(mismatched)} column(s) differ in value, e.g. {mismatched[:5]}."
            )
        else:
            report["checked"].append(
                f"all values match across {report['rows_compared']:,} "
                f"{'rows (entire file)' if full else 'sampled rows'}"
            )
    except Exception as exc:
        report["problems"].append(f"Comparison failed: {exc}")
        return report

    report["ok"] = not report["problems"]
    return report


def _read_csv_head(source: Path, limit: int | None) -> pd.DataFrame:
    """Read the CSV for comparison, using the same reader ``load`` would.

    The reader has to match, or the comparison reports differences that are
    not real: pandas and pyarrow parse the same decimal text into float64
    values that can differ in the last bit, so mixing the two made every float
    column look mismatched while a full comparison passed.
    """
    if limit is None:
        return _read_csv(source, None)
    try:
        import pyarrow as pa
        import pyarrow.csv as pv

        reader = pv.open_csv(
            source,
            read_options=pv.ReadOptions(use_threads=True),
            convert_options=pv.ConvertOptions(strings_can_be_null=True),
        )
        batches = []
        rows = 0
        try:
            for batch in reader:
                batches.append(batch)
                rows += batch.num_rows
                if rows >= limit:
                    break
        finally:
            reader.close()
        if not batches:
            return pd.DataFrame()
        frame = pa.Table.from_batches(batches).to_pandas()
        return frame.head(limit)
    except Exception:
        return pd.read_csv(source, nrows=limit, low_memory=False)


def columns(path: str | Path) -> list[str]:
    """Return the column names of a table without reading any rows."""
    source = Path(path).expanduser()
    if not source.is_file():
        return []
    twin = fast_copy_path(source)
    if twin.is_file() and _twin_matches_source(twin, source):
        try:
            import pyarrow.parquet as pq

            return list(pq.read_schema(twin).names)
        except Exception:
            pass
    try:
        return list(pd.read_csv(source, nrows=0).columns)
    except Exception as exc:
        logger.warning("Could not read the header of %s: %s", source, exc)
        return []


def _should_be_categorical(name: str) -> bool:
    lowered = name.lower()
    if lowered in DEFAULT_CATEGORICAL_HINTS:
        return True
    return lowered.startswith(CATEGORICAL_PREFIXES)


def _apply_categoricals(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert low-cardinality text columns to ``category`` in place."""
    for name in frame.columns:
        if frame[name].dtype == object and _should_be_categorical(str(name)):
            try:
                frame[name] = frame[name].astype("category")
            except (TypeError, ValueError):
                continue
    return frame


def load(
    path: str | Path,
    columns: Sequence[str] | None = None,
    *,
    categoricals: bool = False,
    build_fast_copy: bool = True,
    apply_labels: bool = True,
    progress: Any | None = None,
) -> pd.DataFrame:
    """Read *path*, loading only *columns* when given.

    Reads from the Parquet twin when one is current, which is what makes a
    narrow column request fast rather than merely memory-light.  Falls back to
    the CSV whenever the twin is missing, stale or unreadable, so behaviour is
    unchanged on a machine where the conversion cannot run.

    Passing ``columns=None`` loads every column, which on a multi-gigabyte
    table is the slow path this module exists to avoid; prefer naming what the
    screen needs.

    Labels assigned in the interface live in a small sidecar rather than being
    written back into a multi-gigabyte table on every save, so they are merged
    in here. Pass ``apply_labels=False`` to see the file exactly as it is on
    disk -- which is what committing those labels needs.

    *categoricals* defaults to False and should stay that way for general use.
    Converting text columns to ``category`` saves a great deal of memory, but
    it is a real dtype change, and callers that later assign a value outside
    the existing categories fail with "Cannot setitem on a Categorical with a
    new category". Several screens do exactly that -- assigning a phenotype
    label to an unlabelled cell, for one -- so the conversion is opt-in and
    belongs only where the frame is known to be read-only.
    """
    source = Path(path).expanduser()
    if not source.is_file():
        raise DataIOError(f"{source} does not exist.")

    wanted = list(dict.fromkeys(columns)) if columns else None
    if wanted:
        available = set(_available_columns(source))
        missing = [name for name in wanted if name not in available]
        if missing:
            raise DataIOError(
                f"{source.name} has no column(s): {', '.join(missing)}. "
                f"Available columns include: {', '.join(sorted(available)[:8])}..."
            )

    twin = fast_copy_path(source)
    if fast_copies_enabled():
        if twin.is_file() and _twin_matches_source(twin, source):
            try:
                frame = pd.read_parquet(twin, columns=wanted)
                frame = _with_labels(frame, source, wanted, apply_labels)
                return _apply_categoricals(frame) if categoricals else frame
            except Exception as exc:
                logger.warning("Fast copy unreadable (%s); falling back to CSV.", exc)
        elif build_fast_copy:
            # Missing or stale.  Start building in the background and read the
            # CSV for this call, so the user never waits for a conversion.
            start_fast_copy(source)

    frame = _read_csv(source, wanted)
    frame = _with_labels(frame, source, wanted, apply_labels)
    return _apply_categoricals(frame) if categoricals else frame


def _with_labels(
    frame: pd.DataFrame,
    source: Path,
    wanted: list[str] | None,
    enabled: bool,
) -> pd.DataFrame:
    """Merge any labels assigned in the interface but not yet written back."""
    if not enabled:
        return frame
    try:
        from . import labels

        return labels.apply_to(frame, source, columns=wanted)
    except Exception as exc:
        # A label overlay must never stop data being read.
        logger.warning("Could not apply pending labels for %s: %s", source.name, exc)
        return frame


def _available_columns(source: Path) -> Iterable[str]:
    names = columns(source)
    if not names:
        raise DataIOError(f"Could not read the columns of {source}.")
    return names


def _read_csv(source: Path, wanted: list[str] | None) -> pd.DataFrame:
    """Read a CSV, preferring pyarrow's threaded parser."""
    try:
        import pyarrow.csv as pv

        convert = pv.ConvertOptions(strings_can_be_null=True)
        if wanted:
            convert = pv.ConvertOptions(strings_can_be_null=True, include_columns=wanted)
        table = pv.read_csv(
            source,
            read_options=pv.ReadOptions(use_threads=True),
            convert_options=convert,
        )
        return table.to_pandas(split_blocks=True, self_destruct=True)
    except Exception as exc:
        logger.info("pyarrow CSV reader unavailable (%s); using pandas.", exc)
        return pd.read_csv(source, usecols=wanted, low_memory=False)


__all__ = [
    "DataIOError",
    "clear_fast_copy",
    "columns",
    "ensure_fast_copy",
    "fast_copies_enabled",
    "fast_copy_info",
    "fast_copy_path",
    "is_building",
    "build_error",
    "start_fast_copy",
    "load",
    "verify_fast_copy",
]

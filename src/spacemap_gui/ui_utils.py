import streamlit as st
import os
import base64
import logging
from collections.abc import Iterable, Mapping
from io import BytesIO
from pathlib import Path
from typing import Optional, Callable, Any
from datetime import datetime

# ============================================================================
# st.rerun() POLICY (P2 guidance — 2026-07-20)
# ============================================================================
# The codebase has 150+ st.rerun() calls. Many are necessary (file browser
# navigation, toggling UI panels, applying pending selections before a widget
# renders). The problematic pattern is:
#
#   if st.button("X"):
#       st.session_state[key] = value
#       st.rerun()
#
# which forces a full page re-execution when an on_change callback on the
# source widget would be cheaper. Migrating these requires per-widget UI
# testing and is deferred. In the meantime:
#
# - Prefer on_change callbacks over button + st.rerun() for new code.
# - Wrap heavy, self-contained UI regions in @st.fragment (Streamlit >=1.36)
#   so their reruns don't re-execute the whole page.
# - NEVER call st.rerun() inside a loop or on every render without a guard.
# ============================================================================


# Module-level logger for the GUI layer. Using logging (not print) so that
# silent exception handlers can emit diagnostics without spamming the UI.
_log = logging.getLogger("spacemap_gui")
if not _log.handlers:
    _log.addHandler(logging.StreamHandler())
    _log.setLevel(logging.INFO)


def log_exception(context: str, exc: BaseException, *, show_ui: bool = False) -> None:
    """Log a swallowed exception with context, optionally surfacing in the UI.

    Replaces bare ``except: pass`` / ``except Exception: pass`` patterns so
    that failures are at least recorded in the server log (and optionally
    shown to the user via ``st.warning``).

    Parameters
    ----------
    context:
        Short description of what was being attempted, e.g. ``"loading geojson"``.
    exc:
        The caught exception instance.
    show_ui:
        If ``True``, also call ``st.warning`` with a concise message so the
        user sees something went wrong. Defaults to ``False`` (log only).
    """
    msg = f"[spacemap_gui] {context} failed: {type(exc).__name__}: {exc}"
    _log.warning(msg)
    if show_ui:
        try:
            st.warning(f"{context} failed: {exc}")
        except Exception:
            # st may not be available (e.g. during import); ignore.
            pass


def require_active_project() -> Optional[Path]:
    """Guard for pipeline tabs that need an active project.

    Call at the top of a page script (or ``render()``). If no project is
    loaded, shows a friendly ``st.info`` message, stops execution, and
    returns ``None``. Otherwise returns the active project root as a
    ``Path``.

    Usage::

        from spacemap_gui.ui_utils import require_active_project
        project_root = require_active_project()
        if project_root is None:
            st.stop()  # belt-and-suspenders; require_active_project already stops
    """
    root = st.session_state.get("active_project_root")
    if not root:
        st.info("Please load a Project in the sidebar to view this page.")
        st.stop()
        return None
    return Path(root)


def validate_run_directories(
    input_directories: Mapping[str, str | Path | None],
    output_directory: str | Path | None,
) -> list[str]:
    """Return clear setup errors before a pipeline job reads or writes data.

    Input directories must exist and be readable. The output directory may not
    exist yet, but its nearest existing parent must be writable. Output folders
    cannot be the same as, or nested inside, an input folder because that can
    cause a later run to treat its own results as new input.
    """
    errors: list[str] = []
    resolved_inputs: list[tuple[str, Path]] = []

    for label, directory_value in input_directories.items():
        display_label = label[:1].upper() + label[1:]
        if not directory_value or not str(directory_value).strip():
            errors.append(f"Choose a {label}.")
            continue

        directory = Path(directory_value).expanduser()
        if not directory.exists():
            errors.append(f"{display_label} does not exist: {directory}")
            continue
        if not directory.is_dir():
            errors.append(f"{display_label} is not a directory: {directory}")
            continue
        if not os.access(directory, os.R_OK | os.X_OK):
            errors.append(f"{display_label} cannot be read: {directory}")
            continue
        resolved_inputs.append((label, directory.resolve()))

    if not output_directory or not str(output_directory).strip():
        errors.append("Choose an output directory.")
        return errors

    output_path = Path(output_directory).expanduser()
    if output_path.exists() and not output_path.is_dir():
        errors.append(f"Output path is not a directory: {output_path}")
        return errors

    existing_parent = output_path if output_path.exists() else output_path.parent
    while not existing_parent.exists() and existing_parent != existing_parent.parent:
        existing_parent = existing_parent.parent
    if not existing_parent.exists() or not os.access(existing_parent, os.W_OK | os.X_OK):
        errors.append(f"Cannot create or write to the output directory: {output_path}")
        return errors

    resolved_output = output_path.resolve()
    for label, input_path in resolved_inputs:
        try:
            resolved_output.relative_to(input_path)
        except ValueError:
            continue
        errors.append(
            f"Output directory must be outside the {label}: {resolved_output}"
        )

    return errors


def validate_selected_files(
    selected_paths: Iterable[str | Path], *, item_name: str
) -> list[str]:
    """Return clear errors when a run has no selection or selected files disappeared."""
    paths = [Path(path).expanduser() for path in selected_paths]
    if not paths:
        return [f"Select at least one {item_name} before starting."]

    missing = [path.name for path in paths if not path.is_file()]
    if not missing:
        return []

    listed_names = ", ".join(missing[:3])
    if len(missing) > 3:
        listed_names += f", and {len(missing) - 3} more"
    return [f"Selected {item_name} file(s) no longer exist: {listed_names}"]


def show_preflight_errors(errors: Iterable[str]) -> None:
    """Display setup failures in one consistent, action-oriented message."""
    error_list = list(errors)
    if not error_list:
        return

    st.error("Cannot start this run yet. Fix the following setup issue(s):")
    for error in error_list:
        st.markdown(f"- {error}")


def patch_canvas_image_to_url() -> None:
    """Monkey-patch ``streamlit.elements.image.image_to_url`` if missing.

    ``streamlit-drawable-canvas`` (<=0.9.3) calls ``st_image.image_to_url``,
    which was removed in newer Streamlit versions. This installs a
    compatible replacement that:

    1. Tries Streamlit's ``Runtime.media_file_mgr`` to serve a real
       ``/media/...`` HTTP URL (preferred — works with canvas internals).
    2. Falls back to an inline ``data:image/png;base64,...`` URI.

    Safe to call multiple times — only patches when the attribute is absent.
    Import this and call once at module load in any tab that uses
    ``st_canvas`` (currently Tab 3 and Tab 4).
    """
    import streamlit.elements.image as st_image
    if hasattr(st_image, 'image_to_url'):
        return

    from PIL import Image
    import numpy as np

    def image_to_url(image, width, clamp, channels, output_format, image_id, allow_emoji=False):
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            else:
                return ""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        png_bytes = buffered.getvalue()
        try:
            from streamlit.runtime import Runtime
            runtime = Runtime.instance()
            return runtime.media_file_mgr.add(
                png_bytes,
                "image/png",
                coordinates=str(image_id),
            )
        except Exception:
            encoded = base64.b64encode(png_bytes).decode()
            return f"data:image/png;base64,{encoded}"

    st_image.image_to_url = image_to_url  # type: ignore[attr-defined]


def directory_input_with_browse(
    *,
    label: str,
    text_key: str,
    default_value: str,
    browser_key_prefix: str,
    start_dir_fallback: Optional[str] = None,
    help_text: Optional[str] = None,
    disabled: bool = False,
    on_change: Optional[Callable[..., Any]] = None,
    args: Optional[tuple] = None,
    kwargs: Optional[dict] = None,
) -> str:
    """Render a directory text input with a consistent "Browse" button.

    This component wraps the existing server-side file browser (`render_file_browser`)
    and applies the selection safely by staging it in session state and then
    rerunning before the text input widget is instantiated.

    Parameters
    ----------
    label:
        Label shown for the directory text input.
    text_key:
        Session-state key used by the `st.text_input`.
    default_value:
        Initial value used when `text_key` is not set.
    browser_key_prefix:
        Unique prefix used to namespace internal browser widget keys.
    start_dir_fallback:
        Fallback directory to open in the browser when the current value is invalid.
        If not provided, defaults to `default_value`.
    help_text:
        Optional help text for the text input.
    disabled:
        Whether to disable the text input.
    on_change:
        Optional callback to run when the directory changes.
    args:
        Optional positional arguments for the callback.
    kwargs:
        Optional keyword arguments for the callback.

    Returns
    -------
    str
        The current directory string from the text input.
    """

    # Local import to avoid circular dependencies.
    from spacemap_gui.file_browser import render_file_browser

    show_key = f"{browser_key_prefix}__show"
    pending_key = f"{browser_key_prefix}__pending"
    browser_widget_key = f"{browser_key_prefix}__browser"
    browse_button_key = f"{browser_key_prefix}__browse_button"

    # Set default value in session_state if it's missing or None to avoid rerun logic.
    if st.session_state.get(text_key) is None and default_value:
        st.session_state[text_key] = default_value
        if os.getenv("SPACEMAP_DEBUG") == "1":
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] [UI_UTILS] Auto-populated key '{text_key}' with default '{default_value}'", flush=True)

    # Apply pending selection BEFORE creating the text_input widget.
    if pending_key in st.session_state:
        st.session_state[text_key] = st.session_state.pop(pending_key)
        if on_change is not None:
            on_change(*(args or ()), **(kwargs or {}))

    # Browser UI first so selection can be applied via rerun.
    if st.session_state.get(show_key, False):
        current_dir = str(st.session_state.get(text_key, "") or "")
        start_dir = current_dir if os.path.isdir(current_dir) else (start_dir_fallback or default_value)
        start_dir = start_dir if os.path.isdir(start_dir) else os.getcwd()

        sel_dir = render_file_browser(start_dir, browser_widget_key, select_type="directory")
        if sel_dir == "CANCEL":
            st.session_state[show_key] = False
            st.rerun()
        elif sel_dir:
            st.session_state[pending_key] = sel_dir
            st.session_state[show_key] = False
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] [UI_UTILS] Browser selected folder '{sel_dir}' for key '{text_key}'", flush=True)
            st.rerun()

    # Move label outside to ensure vertical alignment of input and button
    if label:
        st.markdown(f"**{label}**", help=help_text)

    c_text, c_btn = st.columns([3, 1], gap="small")
    with c_text:
        # Prepare kwargs for st.text_input. 
        # Streamlit bug workaround: if a widget is garbage collected, it might ignore 
        # programmatic session_state updates on the next render if value is omitted.
        # By popping it from session_state and passing it as value, we force initialization
        # and avoid the Streamlit Warning about setting both value and session_state.
        text_input_kwargs = {
            "label": label,
            "key": text_key,
            "help": help_text if not label else None,
            "disabled": disabled,
            "label_visibility": "collapsed",
            "on_change": on_change,
            "args": args,
            "kwargs": kwargs,
        }
        
        if text_key in st.session_state:
            text_input_kwargs["value"] = st.session_state.pop(text_key)
        else:
            text_input_kwargs["value"] = default_value

        value = st.text_input(**text_input_kwargs)

    with c_btn:
        if st.button("Browse", key=browse_button_key, disabled=disabled):
            st.session_state[show_key] = True
            st.rerun()

    return str(value)

def st_directory_selector(path_key, label="Select Directory", key_prefix=""):
    """
    A Streamlit component to select a directory on the server.
    
    Args:
        path_key (str): The session_state key where the selected path should be stored.
        label (str): Label for the component.
        key_prefix (str): Unique prefix for widget keys to avoid conflicts.
    
    Returns:
        bool: True if a selection was confirmed, False otherwise.
    """
    # Internal state for navigation (separate from the final selected path)
    nav_key = f"{key_prefix}_nav_path"
    
    if nav_key not in st.session_state:
        # Initialize navigation with the current value of path_key if it exists and is valid
        current_val = st.session_state.get(path_key)
        if current_val and os.path.exists(current_val):
            st.session_state[nav_key] = current_val
        else:
            st.session_state[nav_key] = os.getcwd()
            
    current_path = Path(st.session_state[nav_key])
    
    # Ensure path exists (in case it was deleted or is invalid)
    if not current_path.exists():
        current_path = Path(os.getcwd())
        st.session_state[nav_key] = str(current_path)

    st.markdown(f"**{label}**")
    
    # Navigation Bar
    col1, col2 = st.columns([4, 1])
    with col1:
        st.code(str(current_path), language="text")
    with col2:
        if st.button("⬆️ Up", key=f"{key_prefix}_up"):
            st.session_state[nav_key] = str(current_path.parent)
            st.rerun()

    # Directory Listing
    try:
        # Get directories
        items = [d for d in os.listdir(current_path) if (current_path / d).is_dir()]
        # Filter hidden
        items = [d for d in items if not d.startswith('.')]
        items.sort()
        
        # Navigation Selection
        # We use a selectbox that triggers a rerun when changed
        selected_subdir = st.selectbox(
            "Navigate to...", 
            ["(Stay in current folder)"] + items, 
            key=f"{key_prefix}_subdir_select"
        )
        
        if selected_subdir != "(Stay in current folder)":
            st.session_state[nav_key] = str(current_path / selected_subdir)
            st.rerun()
            
    except PermissionError:
        st.error("Permission Denied")
    except Exception as e:
        st.error(f"Error listing directory: {e}")

    # Confirmation Button
    if st.button(f"Select This Folder", key=f"btn_{key_prefix}_confirm", type="primary"):
        st.session_state[path_key] = str(current_path)
        return True
        
    return False

def st_file_selector(path_key, label="Select File", key_prefix="", extensions=None):
    """
    A Streamlit component to select a file on the server.
    
    Args:
        path_key (str): The session_state key where the selected path should be stored.
        label (str): Label for the component.
        key_prefix (str): Unique prefix for widget keys.
        extensions (list): List of allowed extensions (e.g. ['.csv', '.txt']). None for all.
    """
    nav_key = f"{key_prefix}_nav_path"
    
    if nav_key not in st.session_state:
        current_val = st.session_state.get(path_key)
        if current_val and os.path.exists(current_val):
            # If current value is a file, start in its parent
            p = Path(current_val)
            if p.is_file():
                st.session_state[nav_key] = str(p.parent)
            else:
                st.session_state[nav_key] = str(p)
        else:
            st.session_state[nav_key] = os.getcwd()
            
    current_path = Path(st.session_state[nav_key])
    if not current_path.exists():
        current_path = Path(os.getcwd())
        st.session_state[nav_key] = str(current_path)

    st.markdown(f"**{label}**")
    
    # Navigation
    col1, col2 = st.columns([4, 1])
    with col1:
        st.code(str(current_path), language="text")
    with col2:
        if st.button("⬆️ Up", key=f"{key_prefix}_up"):
            st.session_state[nav_key] = str(current_path.parent)
            st.rerun()

    try:
        # List Dirs and Files
        all_items = os.listdir(current_path)
        dirs = sorted([d for d in all_items if (current_path / d).is_dir() and not d.startswith('.')])
        files = sorted([f for f in all_items if (current_path / f).is_file() and not f.startswith('.')])
        
        if extensions:
            files = [f for f in files if any(f.endswith(ext) for ext in extensions)]

        # Directory Navigation
        selected_subdir = st.selectbox(
            "Navigate Folder...", 
            ["(Stay in current folder)"] + dirs, 
            key=f"{key_prefix}_subdir_select"
        )
        
        if selected_subdir != "(Stay in current folder)":
            st.session_state[nav_key] = str(current_path / selected_subdir)
            st.rerun()
            
        # File Selection
        selected_file = st.selectbox(
            "Select File...", 
            ["(No file selected)"] + files,
            key=f"{key_prefix}_file_select"
        )
        
        if selected_file != "(No file selected)":
            if st.button("Confirm Selection", key=f"btn_{key_prefix}_confirm", type="primary"):
                st.session_state[path_key] = str(current_path / selected_file)
                return True
                
    except Exception as e:
        st.error(f"Error: {e}")
        
    return False

def render_heavy_tab_gate(tab_key: str, label: str) -> bool:
    """Render a gate button for heavy tabs to defer loading."""
    if "_spacemap_loaded_top_tabs" not in st.session_state:
        st.session_state["_spacemap_loaded_top_tabs"] = {}
    loaded = bool(st.session_state.get("_spacemap_loaded_top_tabs", {}).get(tab_key, False))
    if loaded:
        return True
    
    st.info(f"{label} is deferred for faster app startup.")
    if st.button(f"Load {label}", key=f"{tab_key}_load_top"):
        st.session_state["_spacemap_loaded_top_tabs"][tab_key] = True
        from datetime import datetime
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TAB: Loading deferred tab '{label}'", flush=True)
        st.rerun()
    return False


def file_signature(path) -> tuple[int, int]:
    """Return a cheap change token for a file: (mtime_ns, size).

    ``st.cache_data`` keys on argument values, so a function cached only on a
    path keeps serving the first version it read - editing that file in QuPath,
    Excel or an earlier pipeline step has no effect until the whole session is
    restarted. Passing this token as an extra argument ties the cache entry to
    the file's content instead of just its name.
    """
    try:
        status = os.stat(str(path))
    except OSError:
        return (0, -1)
    return (int(status.st_mtime_ns), int(status.st_size))


def clamp_numeric_widget_state(
    key: str,
    min_value,
    max_value,
    *,
    fallback=None,
):
    """Pull a remembered numeric widget value back inside data-derived bounds.

    Streamlit raises ``StreamlitValueAboveMaxError`` when a widget is created
    with a ``min_value``/``max_value`` that excludes the value it already holds
    in session state. Any widget whose bounds come from the loaded data hits this
    the moment the data changes: switching to a smaller unified CSV leaves the
    previous file's subsample size behind, and the widget refuses to render.

    Clamping the ``value=`` argument is *not* enough -- once a widget key exists,
    the stored value wins over ``value=`` -- so the stored value itself has to be
    corrected, before the widget is instantiated.

    Returns the clamped number, so the caller can pass it as ``value=`` too.
    """
    import streamlit as st

    is_integer = isinstance(min_value, int) and isinstance(max_value, int)
    try:
        low, high = float(min_value), float(max_value)
    except (TypeError, ValueError):
        return fallback
    if high < low:
        # Degenerate range (for example an empty table): nothing can satisfy it,
        # so drop the stale entry and let the caller's own default apply.
        st.session_state.pop(key, None)
        return fallback

    stored = st.session_state.get(key)
    candidate = stored if stored is not None else fallback
    if candidate is None:
        return None
    try:
        numeric = float(candidate)
    except (TypeError, ValueError):
        st.session_state.pop(key, None)
        return fallback

    clamped = min(max(numeric, low), high)
    if is_integer:
        clamped = int(round(clamped))
    if key in st.session_state and st.session_state[key] != clamped:
        st.session_state[key] = clamped
    return clamped


def _format_bytes(value: float) -> str:
    """Human-readable size for the fast-copy panel."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit in ("B", "KB") else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def render_fast_copy_controls(csv_path: str | None, *, key_prefix: str = "fastcopy") -> None:
    """Show and manage the Parquet fast copy that mirrors a large CSV.

    SPACEMAP keeps a columnar copy beside each large CSV and reads columns from
    it, because a CSV must be parsed from the beginning to reach any column: a
    two-column question on a 13 GB table takes minutes from the CSV and about a
    second from the copy.

    The copy is created automatically in the background the first time a large
    CSV is used, and rebuilt whenever the CSV changes, so the two are never out
    of step.  Reads fall back to the CSV whenever the copy is missing, stale or
    still building, so nothing ever waits on it.  The CSV is never modified and
    the copy can be deleted at any time.
    """
    import streamlit as st

    if not csv_path:
        return
    try:
        from spacemap import dataio
    except Exception:
        return

    try:
        info = dataio.fast_copy_info(csv_path)
        building = dataio.is_building(csv_path)
        error = dataio.build_error(csv_path)
    except Exception:
        return
    if not info.get("source_bytes"):
        return
    if not info["enabled"]:
        st.caption("Fast loading is switched off (`SPACEMAP_FAST_COPY=off`).")
        return
    if not info["worth_building"] and not info["exists"]:
        return

    # Labels assigned in the interface live in a sidecar until they are folded
    # into the CSV, because rewriting a multi-gigabyte table on every save took
    # minutes. Surface how many are waiting, and let the user commit them.
    try:
        from spacemap import labels

        pending = labels.pending_count(csv_path)
        pending_columns = labels.pending_columns(csv_path)
    except Exception:
        pending, pending_columns = 0, []

    if pending:
        with st.container():
            st.info(
                f"**{pending} label(s) saved** in "
                f"{', '.join(f'`{name}`' for name in pending_columns)}, not yet written "
                "into the CSV. They are applied every time the data is read, so the "
                "interface already shows them."
            )
            left, right = st.columns(2)
            if left.button("Write labels into the CSV", key=f"{key_prefix}_commit", width="stretch"):
                status = st.empty()
                with st.spinner("Writing the CSV and refreshing the fast copy..."):
                    try:
                        report = labels.commit(csv_path, progress=lambda m: status.caption(m))
                    except Exception as exc:
                        status.empty()
                        st.error(f"Could not write the labels: {exc}")
                    else:
                        status.empty()
                        st.success(
                            f"Wrote {report['applied']} label(s) into "
                            f"{', '.join(report['columns'])}."
                        )
                        st.rerun()
            if right.button("Discard saved labels", key=f"{key_prefix}_discard", width="stretch"):
                labels.clear(csv_path)
                st.rerun()

    if building:
        icon, label = "⏳", "Fast loading: preparing in background"
    elif info["current"]:
        icon, label = "✅", "Fast loading: on"
    elif error:
        icon, label = "❌", "Fast loading: could not be prepared"
    elif info["exists"]:
        icon, label = "⚠️", "Fast loading: out of date, refreshing"
    else:
        icon, label = "•", "Fast loading: not prepared yet"

    with st.expander(f"{icon} {label}", expanded=bool(error)):
        st.caption(
            f"CSV {_format_bytes(info['source_bytes'])}"
            + (f" · fast copy {_format_bytes(info['twin_bytes'])}" if info["exists"] else "")
            + f" · {info['twin']}"
        )

        if building:
            st.info(
                "Building the fast copy in the background. Everything keeps "
                "working from the CSV meanwhile; reads get much quicker once it "
                "finishes. Reopen this panel to check."
            )
        elif info["current"]:
            st.success(
                "Columns are read from the fast copy. It is rebuilt "
                "automatically whenever the CSV changes, so the two always agree."
            )
        elif error:
            st.error(f"{error} SPACEMAP is reading the CSV instead, so nothing is broken.")
        elif info["exists"]:
            st.warning("The CSV changed. The copy refreshes on next use.")
        else:
            st.info(
                "A fast copy will be prepared automatically the first time this "
                "table is read. You can start it now instead."
            )

        left, middle, right = st.columns(3)
        build_label = "Rebuild now" if info["exists"] else "Prepare now"
        if left.button(build_label, key=f"{key_prefix}_build", width="stretch", disabled=building):
            status = st.empty()
            with st.spinner("Preparing the fast copy..."):
                built = dataio.ensure_fast_copy(
                    csv_path, force=True, progress=lambda m: status.caption(m)
                )
            status.empty()
            if built:
                st.success("Fast copy ready.")
            else:
                st.error("Could not prepare the fast copy; the CSV will be used.")
            st.rerun()

        if middle.button(
            "Check it matches", key=f"{key_prefix}_verify", width="stretch",
            disabled=building or not info["current"],
        ):
            with st.spinner("Comparing the fast copy against the CSV..."):
                report = dataio.verify_fast_copy(csv_path)
            if report["ok"]:
                st.success("The fast copy matches the CSV: " + "; ".join(report["checked"]) + ".")
            else:
                for problem in report["problems"]:
                    st.error(problem)

        if info["exists"] and right.button(
            "Delete copy", key=f"{key_prefix}_delete", width="stretch", disabled=building
        ):
            dataio.clear_fast_copy(csv_path)
            st.rerun()

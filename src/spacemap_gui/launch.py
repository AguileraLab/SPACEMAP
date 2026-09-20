"""The ``spacemap`` command: start the Streamlit interface.

Installing the package puts this on the PATH, so a user runs ``spacemap``
rather than remembering a Streamlit invocation and a path into site-packages.

It also applies the numeric thread caps the pipeline needs before NumPy is
imported.  That is not cosmetic: the OpenBLAS shipped with NumPy is built with
a fixed table of 128 thread buffers, and on a machine with more cores than that
scikit-learn's k-means opens one OpenMP thread per core, every one of which
calls into BLAS.  OpenBLAS then exhausts the table, prints "Program is
Terminated. Because you tried to allocate too many memory regions", returns a
null buffer and segfaults its caller.  Capping the pools keeps that from
happening and costs nothing on a smaller machine.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Combined OpenMP and BLAS pools stay well under the OpenBLAS limit at this
#: value, which is ample for the numeric work the pipeline does.
THREAD_CAP = 32

_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _apply_thread_caps() -> None:
    """Cap numeric thread pools unless the user already chose a value."""
    cores = os.cpu_count() or 8
    cap = str(min(cores, THREAD_CAP))
    for name in _THREAD_VARIABLES:
        os.environ.setdefault(name, cap)


def _apply_runtime_defaults() -> None:
    """Environment the GUI expects, set only when it is not already set."""
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # A site proxy must never intercept traffic to a model server running on
    # this machine.
    existing = os.environ.get("NO_PROXY", "")
    if "127.0.0.1" not in existing:
        os.environ["NO_PROXY"] = ",".join(filter(None, [existing, "127.0.0.1,localhost"]))
        os.environ["no_proxy"] = os.environ["NO_PROXY"]


def app_path() -> Path:
    """Absolute path to the Streamlit entry script inside the package."""
    return Path(__file__).resolve().parent / "app.py"


def main(argv: list[str] | None = None) -> int:
    """Run the SPACEMAP interface.  Extra arguments are passed to Streamlit."""
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] in {"-h", "--help"}:
        print(__doc__.strip())
        print(
            "\nUsage: spacemap [--port PORT] [streamlit options]\n\n"
            "Common options:\n"
            "  --port PORT            Serve on this port (default 8501).\n"
            "  --server.headless      Do not open a browser automatically.\n"
            "\nEnvironment:\n"
            "  SPACEMAP_HOME          Where settings, models and logs are kept.\n"
            "  SPACEMAP_ASSISTANT     Set to 'off' to disable the AI assistant.\n"
            "  SPACEMAP_DATA_DIR      Default data location used by the notebooks.\n"
        )
        return 0

    if argv and argv[0] == "--version":
        from spacemap import __version__

        print(f"SPACEMAP {__version__}")
        return 0

    _apply_thread_caps()
    _apply_runtime_defaults()

    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError:
        print(
            "Streamlit is not installed in this environment.\n"
            "Install SPACEMAP with:  pip install spacemap-ai",
            file=sys.stderr,
        )
        return 1

    # Translate the friendly --port form into Streamlit's own flag.
    arguments: list[str] = []
    index = 0
    while index < len(argv):
        if argv[index] == "--port" and index + 1 < len(argv):
            arguments += [f"--server.port={argv[index + 1]}"]
            index += 2
            continue
        arguments.append(argv[index])
        index += 1

    if not any(argument.startswith("--server.port") for argument in arguments):
        arguments.append("--server.port=8501")
    if not any(argument.startswith("--browser.gatherUsageStats") for argument in arguments):
        arguments.append("--browser.gatherUsageStats=false")

    sys.argv = ["streamlit", "run", str(app_path()), *arguments]
    return int(streamlit_cli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())

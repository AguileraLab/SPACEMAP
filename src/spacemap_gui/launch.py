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

When the chosen port is already in use on this machine (another SPACEMAP, or another
user's app), the next free port is used and the running app is left alone.
"""

from __future__ import annotations

import os
import socket
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
    """Absolute path to the Streamlit entry script inside the package.

    ``server.py`` runs the same interface and also serves the image viewer's
    routes on the same port. Streamlit releases without ``st.App`` get ``app.py``
    and the viewer starts its routes on a second port instead.
    """
    here = Path(__file__).resolve().parent
    try:
        import streamlit

        if hasattr(streamlit, "App"):
            return here / "server.py"
    except ImportError:
        pass
    return here / "app.py"


def available_port(start=8502, address=""):
    """Check an actual bind on this node, including ports owned by other users."""
    if not 1 <= start <= 65535:
        raise ValueError("The server port must be between 1 and 65535.")
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    for port in range(start, min(start + 100, 65536)):
        with socket.socket(family, socket.SOCK_STREAM) as listener:
            try:
                listener.bind((address, port))
            except OSError as error:
                import errno
                if error.errno == errno.EADDRINUSE:
                    continue
                raise
            return port
    raise ValueError(f"No available port in {start}–{min(start + 99, 65535)}.")


def command(entry, arguments):
    args, port, address = [], os.environ.get("STREAMLIT_SERVER_PORT", "8502"), os.environ.get("STREAMLIT_SERVER_ADDRESS", "")
    iterator = iter(arguments)
    for arg in iterator:
        if arg == "--server.port":
            port = next(iterator, "")
        elif arg.startswith("--server.port="):
            port = arg.split("=", 1)[1]
        else:
            args.append(arg)
            if arg == "--server.address":
                address = next(iterator, "")
                args.append(address)
            elif arg.startswith("--server.address="):
                address = arg.split("=", 1)[1]
    requested = int(port)
    chosen = available_port(requested, address)
    if chosen != requested:
        print(f"Port {requested} is in use on {socket.gethostname()}; using {chosen}. The existing app stays running.", flush=True)
    print(f"Starting SPACEMAP on {socket.gethostname()}:{chosen} (local URL: http://localhost:{chosen}).", flush=True)
    return [sys.executable, "-m", "streamlit", "run", entry, f"--server.port={chosen}", *args]


def main(argv: list[str] | None = None) -> int:
    """Run the SPACEMAP interface.  Extra arguments are passed to Streamlit."""
    argv = list(sys.argv[1:] if argv is None else argv)

    # run_interface.sh names the entry script first; the installed command does not.
    entry = str(app_path())
    if argv and argv[0].endswith(".py") and Path(argv[0]).is_file():
        entry = argv.pop(0)
        os.environ.setdefault("STREAMLIT_SERVER_PORT", "8502")  # the script's usual port

    if argv and argv[0] in {"-h", "--help"}:
        print(__doc__.strip())
        print(
            "\nUsage: spacemap [--port PORT] [streamlit options]\n"
            "       spacemap doctor        Check this environment and say what to fix.\n"
            "       spacemap share-project FOLDER\n"
            "                              Give the folder's group read and write on your files in it.\n\n"
            "Common options:\n"
            "  --port PORT            Serve on this port (default 8501; the next free one if it is busy).\n"
            "  --server.headless      Do not open a browser automatically.\n"
            "\nEnvironment:\n"
            "  SPACEMAP_HOME          Where settings, models and logs are kept.\n"
            "  SPACEMAP_ASSISTANT     Set to 'off' to disable the AI assistant.\n"
            "  SPACEMAP_DATA_DIR      Default data location used by the notebooks.\n"
        )
        return 0

    if argv and argv[0] == "doctor":
        from spacemap.doctor import main as doctor

        return doctor(argv[1:])

    if argv and argv[0] == "share-project":
        from spacemap.sharing import repair

        if len(argv) != 2 or not Path(argv[1]).is_dir():
            print("Usage: spacemap share-project FOLDER", file=sys.stderr)
            return 2
        done = repair(argv[1], budget_seconds=3600)
        print(f"Checked {done['checked']:,} files and folders; gave the group access on {done['changed']:,} of yours."
              + ("" if done["complete"] else " Stopped after an hour: run it again to finish."))
        print("Each collaborator runs this (or simply opens the project in SPACEMAP) for the files they own.")
        return 0

    if argv and argv[0] == "--version":
        from spacemap import __version__

        print(f"SPACEMAP {__version__}")
        return 0

    _apply_thread_caps()
    _apply_runtime_defaults()
    from spacemap.sharing import apply_group_umask

    apply_group_umask()  # files SPACEMAP writes: owner and group read and write (shared lab projects)

    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError:
        print(
            "Streamlit is not installed in this environment.\n"
            "From the cloned SPACEMAP folder, run:  python -m pip install -e .",
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

    if not any(argument.startswith("--browser.gatherUsageStats") for argument in arguments):
        arguments.append("--browser.gatherUsageStats=false")
    os.environ.setdefault("STREAMLIT_SERVER_PORT", "8501")
    try:
        found = command(entry, arguments)  # a free port: the asked one, else the next
    except (ValueError, OSError) as error:
        print(f"Could not choose a port: {error}", file=sys.stderr)
        return 1

    sys.argv = ["streamlit", *found[3:]]
    return int(streamlit_cli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())

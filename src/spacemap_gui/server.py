"""Start SPACEMAP with the image viewer's data routes on the same port.

    streamlit run src/spacemap_gui/server.py

Serves the interface (``app.py``) exactly as before, plus the viewer's tile and
data routes under ``/spacemap-viewer/api``. Because both share one port, nothing
extra has to be forwarded when SPACEMAP runs on a cluster node.
"""

import os
import socket
import sys
import threading
import time
from pathlib import Path

# This checkout's code first, as app.py does: an older SPACEMAP installed in the
# environment (an editable install pointing at another folder) must not be the
# one imported here.
_SOURCE_ROOT = str(Path(__file__).resolve().parents[1])
if _SOURCE_ROOT in sys.path:
    sys.path.remove(_SOURCE_ROOT)
sys.path.insert(0, _SOURCE_ROOT)

import streamlit as st  # noqa: E402
from starlette.routing import Mount  # noqa: E402

from spacemap_gui.viewer import api  # noqa: E402

api.MOUNTED = True


def _announce_when_ready() -> None:
    """Print where to open SPACEMAP once it is listening.

    Streamlit prints its "You can now view your Streamlit app" addresses only
    for plain scripts; for an ``st.App`` it prints nothing, which looks like a
    hang. This prints the same addresses when the port answers.
    """
    port = int(st.get_option("server.port") or 8501)
    address = str(st.get_option("server.address") or "")
    base = str(st.get_option("server.baseUrlPath") or "").strip("/")
    path = f"/{base}" if base else ""
    probe = address if address not in ("", "0.0.0.0", "::") else "127.0.0.1"
    for _ in range(1200):  # up to 10 minutes on a slow file system
        try:
            with socket.create_connection((probe, port), timeout=1):
                break
        except OSError:
            time.sleep(0.5)
    else:
        print(f"SPACEMAP did not start listening on port {port}. Is another program using it?", flush=True)
        return
    host = socket.gethostname()
    try:
        ip = socket.gethostbyname(host)
    except OSError:
        ip = ""
    lines = ["", "  SPACEMAP is running. Open it in your browser:", "",
             f"  Local URL:   http://localhost:{port}{path}",
             f"  Network URL: http://{host}:{port}{path}"]
    if ip and ip != host:
        lines.append(f"               http://{ip}:{port}{path}")
    lines += ["", "  On a cluster, forward this port from the node above (for example",
              f"  ssh -L {port}:{host}:{port} <login node>) and open the Local URL.", ""]
    print("\n".join(lines), flush=True)
    try:  # so Studio can list this user's running servers and stop one left behind
        from spacemap_gui.studio import servers

        servers.register(port, _SOURCE_ROOT)
        api.SERVER_PORT = port
    except Exception:  # noqa: BLE001 - never stop SPACEMAP over the list
        pass


_STOP_GRACE_SECONDS = 5


def _stop_promptly() -> None:
    """Make Ctrl+C stop SPACEMAP within a few seconds.

    The first Ctrl+C asks the server to shut down cleanly, which waits for any
    running page update to finish; a long analysis could hold it for minutes.
    After a short grace period the process exits anyway. Project files are
    written to a temporary file and then swapped in, so stopping never leaves
    one half-written.
    """
    try:
        import uvicorn
    except ImportError:
        return
    original = uvicorn.Server.handle_exit
    if getattr(original, "_spacemap_prompt", False):
        return

    def handle_exit(self, sig, frame):
        original(self, sig, frame)
        if getattr(self, "_spacemap_stopping", False):
            return
        self._spacemap_stopping = True
        print(f"\nStopping SPACEMAP (at most {_STOP_GRACE_SECONDS} s)...", flush=True)

        def force_exit() -> None:
            time.sleep(_STOP_GRACE_SECONDS)
            os._exit(0)

        threading.Thread(target=force_exit, name="spacemap-stop", daemon=True).start()

    handle_exit._spacemap_prompt = True
    uvicorn.Server.handle_exit = handle_exit


_stop_promptly()

if not os.environ.get("_SPACEMAP_ANNOUNCED"):
    os.environ["_SPACEMAP_ANNOUNCED"] = "1"
    print(f"Starting SPACEMAP on port {st.get_option('server.port')} (this can take a minute)...", flush=True)
    threading.Thread(target=_announce_when_ready, name="spacemap-announce", daemon=True).start()

app = st.App(
    str(Path(__file__).resolve().with_name("app.py")),
    routes=[Mount(api.MOUNT_PATH, app=api.app)],
)

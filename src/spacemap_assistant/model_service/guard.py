"""Run the local model server, and stop it when the SPACEMAP that started it stops.

The model server is started in its own session so that closing a browser tab
does not stop it; without this guard it also outlived SPACEMAP itself, and a
server left on a node kept its port and its GPU memory. The guard starts the
server, checks every few seconds that SPACEMAP (the process that started it,
recognised by its process number and start time) is still running, and stops
the server when it is not.

Usage: python guard.py <SPACEMAP pid> <SPACEMAP start time> -- <server command ...>
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

CHECK_SECONDS = 5


def _alive(pid: int, started: float) -> bool:
    try:
        import psutil
    except ImportError:  # without psutil only the process number can be checked
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    try:
        process = psutil.Process(pid)
        # A reused process number belongs to a process that started later.
        return abs(process.create_time() - started) < 2.0 and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def main(argv: list[str]) -> int:
    if "--" not in argv or len(argv) < 4:
        print(__doc__, file=sys.stderr)
        return 2
    parent, started = int(argv[0]), float(argv[1])
    child = subprocess.Popen(argv[argv.index("--") + 1 :])

    def stop(*_args) -> None:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while child.poll() is None:
        if not _alive(parent, started):
            print("SPACEMAP stopped, so the model server is stopping too.", flush=True)
            stop()
        time.sleep(CHECK_SECONDS)
    return child.returncode or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

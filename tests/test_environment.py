"""The environment SPACEMAP runs in: its own modules must not break the libraries it shares."""

from __future__ import annotations

import itertools
import subprocess
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src"


def test_spacemap_keeps_the_built_in_pairwise():
    built_in = itertools.pairwise
    import spacemap.utils  # noqa: F401  (used to replace it for palom)

    assert itertools.pairwise is built_in


def test_the_model_server_loads_after_spacemap():
    # Replacing itertools.pairwise made torch's compiler refuse to load, and with
    # it transformers 5 and the assistant's model server. Run in a fresh process,
    # in the order the assistant uses: SPACEMAP first, then the model libraries.
    script = ("import spacemap.utils, itertools; "
              "import spacemap_assistant.model_service.server; "
              "print(type(itertools.pairwise).__name__)")
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600,
                          env={**__import__("os").environ, "PYTHONPATH": str(SOURCE)})
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "type"

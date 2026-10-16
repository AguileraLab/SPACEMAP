"""Account identifiers must not bypass the publication checker through excluded formats."""

from pathlib import Path
import runpy

import pytest

CHECKER = Path(__file__).resolve().parents[1] / "scripts" / "check_no_local_paths.py"


@pytest.mark.parametrize("relative", ["docs/LEDGER.md", "notebooks/demo.ipynb", "frontend/build/bundle.js", "asset.bin"])
def test_account_identifiers_are_blocked_in_every_publishable_format(tmp_path, monkeypatch, capsys, relative):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    account = "s" + str(120000 + 3456)
    path.write_bytes(b"\x00" + ("/work/" + account + "/study").encode())
    check = runpy.run_path(str(CHECKER))["main"]
    monkeypatch.setitem(check.__globals__, "_tracked_files", lambda: [Path(relative)])
    monkeypatch.chdir(tmp_path)
    assert check() == 1
    assert "personal account identifier" in capsys.readouterr().out


def test_portable_directory_example_is_allowed(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("SPACEMAP_HOME=/path/to/your/workspace")
    check = runpy.run_path(str(CHECKER))["main"]
    monkeypatch.setitem(check.__globals__, "_tracked_files", lambda: [Path("README.md")])
    monkeypatch.chdir(tmp_path)
    assert check() == 0


@pytest.mark.parametrize("relative", ["frontend/studio.js", "frontend/build/bundle.js", "docs/install.md"])
@pytest.mark.parametrize("private_text", ["Nucleus" + "C" + str(140), "/cm/" + "shared/apps/python"])
def test_internal_hosts_and_install_paths_are_blocked(tmp_path, monkeypatch, capsys, relative, private_text):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(private_text)
    check = runpy.run_path(str(CHECKER))["main"]
    monkeypatch.setitem(check.__globals__, "_tracked_files", lambda: [Path(relative)])
    monkeypatch.chdir(tmp_path)
    assert check() == 1
    assert "Machine-specific paths found" in capsys.readouterr().out

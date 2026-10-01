"""The public change ledger must not reintroduce private paths from old commits."""

from pathlib import Path
import runpy

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "update_ledger.py"


def test_old_machine_paths_are_redacted_in_tables_and_summaries():
    ledger = runpy.run_path(str(SCRIPT))
    account = "s" + str(120000 + 3456)
    path = str(Path("/", "storage", "work", "lab", account, "project"))
    commit = ledger["Commit"](
        full_hash="a" * 40, short_hash="a" * 7, author="Colleague", email="colleague@example.org",
        date="2026-10-01T10:00:00+00:00", subject="fix: handle alias " + path,
        body="Why: the account " + account + " opened the same folder twice.", files=["src/example.py"],
    )
    rendered = ledger["render"]([commit])
    assert path not in rendered and account not in rendered
    assert "[local directory]" in rendered and "[account]" in rendered
    assert "opened the same folder twice" in rendered
    assert "src/example.py" in rendered

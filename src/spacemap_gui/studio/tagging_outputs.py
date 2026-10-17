"""Stage tagged tables before a short, protected publication transaction."""

import json
import os
from pathlib import Path
import shutil
import tempfile

from spacemap_workflows.leiden_job import write_json


class StagedTables:
    def __init__(self, output, folder=None):
        self.folder = Path(folder).resolve() if folder else None
        self.directory = Path(tempfile.mkdtemp(prefix=".studio-tagging-", dir=output))
        self.pending = []
        if self.folder:
            self.folder.mkdir(parents=True, exist_ok=True)
            (self.directory / ".owner").write_text(str(self.folder))
            write_json(self.folder / "tagging-staging.json", {"directory": str(self.directory.resolve())})

    def stage(self, frame, target, write):
        target = Path(target)
        if any(existing == target for _, existing in self.pending):
            raise ValueError(f"Two tables would write the same output: {target.name}")
        temporary = self.directory / f"{len(self.pending)}{target.suffix}"
        write(frame, temporary)
        self.pending.append((temporary, target))

    def publish(self, report):
        if not self.pending:
            return []
        report("Saving tagged tables…", .99)
        protect = getattr(report, "protect_output", None)
        if protect:
            protect()
        backups, published = {}, []
        try:
            # Same filesystem: links retain complete old files without copying
            # gigabytes or removing them from view before replacement.
            for number, (_, target) in enumerate(self.pending):
                if target.exists():
                    backup = self.directory / f"previous-{number}"
                    os.link(target, backup)
                    backups[target] = backup
            for temporary, target in self.pending:
                os.replace(temporary, target)
                published.append(target)
        except BaseException:
            for target in reversed(published):
                if target in backups:
                    os.replace(backups[target], target)
                else:
                    target.unlink(missing_ok=True)
            raise
        return list(map(str, published))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        shutil.rmtree(self.directory)
        if self.folder:
            (self.folder / "tagging-staging.json").unlink(missing_ok=True)


def cleanup_staging(folder):
    """Remove only this exited worker's marked scratch directory after a kill."""
    folder = Path(folder).resolve()
    manifest = folder / "tagging-staging.json"
    if not manifest.exists():
        return
    try:
        directory = Path(json.loads(manifest.read_text())["directory"])
        if (directory.name.startswith(".studio-tagging-") and not directory.is_symlink()
                and (directory / ".owner").read_text() == str(folder)):
            shutil.rmtree(directory)
            manifest.unlink(missing_ok=True)
    except (OSError, ValueError, KeyError):
        # An unavailable output filesystem must not prevent job reconciliation.
        pass

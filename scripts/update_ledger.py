#!/usr/bin/env python3
"""Generate ``docs/LEDGER.md``: every change ever made, who made it, and why.

Git already records the full history. What it does not do is present it in a
form a reviewer, an auditor or a new lab member can read without knowing Git.
This script renders that history into one Markdown file:

* every commit, newest first,
* who authored it and when,
* the reason, taken from the ``Why:`` trailer or the commit body,
* which files it touched, and how many lines moved.

Run it after committing changes, then commit the ledger in its own commit:

    python scripts/update_ledger.py            # rewrite docs/LEDGER.md
    python scripts/update_ledger.py --check    # fail if it is out of date

``--check`` is what CI runs, so a pull request that forgets to regenerate the
ledger is caught before it lands.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

LEDGER_PATH = Path("docs/LEDGER.md")

#: Field separator unlikely to appear in a commit message.
UNIT = "\x1f"
RECORD = "\x1e"

GIT_FORMAT = UNIT.join(["%H", "%h", "%an", "%ae", "%aI", "%s", "%b"]) + RECORD

HEADER = """# Change ledger

Every change ever made to this repository, newest first: what changed, who made
it, and why.

This file is generated from the Git history by `scripts/update_ledger.py`. Do
not edit it by hand -- run the script after committing changes. The
"Why" column comes from the `Why:` line in each commit message; see
[CONTRIBUTING.md](../CONTRIBUTING.md) for how to write one.

Local machine paths, account identifiers and site-specific cluster names are
redacted in this public rendering. The original Git history is unchanged.

"""


@dataclass
class Commit:
    """One recorded change."""

    full_hash: str
    short_hash: str
    author: str
    email: str
    date: str
    subject: str
    body: str
    files: list[str] = field(default_factory=list)
    insertions: int = 0
    deletions: int = 0

    @property
    def why(self) -> str:
        """The stated reason for the change.

        A ``Why:`` trailer is preferred. Failing that, the first paragraph of
        the body is used, because a commit that explains itself in prose is
        still better than nothing.
        """
        lines = self.body.splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.lower().startswith("why:"):
                continue
            # A reason often wraps across several lines. Keep reading until a
            # blank line or the next trailer.
            collected = [stripped[4:].strip()]
            for continuation in lines[index + 1:]:
                text = continuation.strip()
                if not text or re.match(r"^[A-Z][A-Za-z-]+:\s", text):
                    break
                collected.append(text)
            return " ".join(part for part in collected if part)
        paragraph: list[str] = []
        for line in self.body.strip().splitlines():
            if not line.strip():
                break
            if line.strip().lower().startswith(("co-authored-by:", "signed-off-by:")):
                continue
            paragraph.append(line.strip())
        return " ".join(paragraph) or "_not recorded_"

    @property
    def day(self) -> str:
        return self.date[:10]


def _git(*args: str) -> str:
    """Run a git command and return its output."""
    try:
        result = subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        )
    except FileNotFoundError:
        sys.exit("git is not installed, so the ledger cannot be generated.")
    except subprocess.CalledProcessError as exc:
        sys.exit(f"git {' '.join(args)} failed:\n{exc.stderr.strip()}")
    return result.stdout


def read_commits() -> list[Commit]:
    """Read the full history, with per-commit file statistics."""
    raw = _git("log", "--no-merges", f"--pretty=format:{GIT_FORMAT}", "--date=iso-strict")
    commits: list[Commit] = []
    for chunk in raw.split(RECORD):
        chunk = chunk.strip("\n")
        if not chunk.strip():
            continue
        parts = chunk.split(UNIT)
        if len(parts) < 7:
            continue
        commit = Commit(
            full_hash=parts[0],
            short_hash=parts[1],
            author=parts[2],
            email=parts[3],
            date=parts[4],
            subject=parts[5],
            body=parts[6],
        )
        commits.append(commit)

    for commit in commits:
        stats = _git("show", "--numstat", "--format=", commit.full_hash)
        for line in stats.splitlines():
            fields = line.split("\t")
            if len(fields) != 3:
                continue
            added, removed, path = fields
            commit.files.append(path)
            # Binary files report "-" instead of a count.
            commit.insertions += int(added) if added.isdigit() else 0
            commit.deletions += int(removed) if removed.isdigit() else 0

    # A commit that only rewrites the ledger is bookkeeping, not a change to
    # the project. Excluding it also makes this script a fixed point: writing
    # the ledger and committing it does not make the ledger out of date again,
    # so the --check in CI is stable.
    return [commit for commit in commits if not _is_ledger_only(commit)]


def _is_ledger_only(commit: Commit) -> bool:
    """True when a commit touches nothing but the generated ledger."""
    ledger = LEDGER_PATH.as_posix()
    return bool(commit.files) and all(path == ledger for path in commit.files)


def _public_text(value: str) -> str:
    """Keep machine details from old commits out of the published ledger."""
    value = re.sub(
        r"(?<![\w:])/(?:[A-Za-z0-9_.-]+/)*(?:work|home[0-9]*|Users|scratch)(?=/|\b)(?:/[^\s|`\"<>(),;]*)?",
        "[local directory]", value,
    )
    value = re.sub(r"s[0-9]{6}", "[account]", value, flags=re.IGNORECASE)
    return re.sub(r"\b[A-Za-z]+hpc\b", "the cluster", value, flags=re.IGNORECASE)


def _escape(value: str) -> str:
    """Make a value safe inside a Markdown table cell."""
    return _public_text(value).replace("|", "\\|").replace("\n", " ").strip()


def render(commits: list[Commit]) -> str:
    """Build the ledger document."""
    lines = [HEADER]
    if not commits:
        lines.append("_No commits recorded yet._\n")
        return "".join(lines)

    authors = sorted({commit.author for commit in commits})
    touched = len({path for commit in commits for path in commit.files})
    lines.append(
        f"**{len(commits)} changes** by **{len(authors)} "
        f"{'contributor' if len(authors) == 1 else 'contributors'}**, "
        f"touching **{touched} files**. "
        f"Latest: `{commits[0].short_hash}` on {commits[0].day}.\n\n"
    )

    current_month = ""
    for commit in commits:
        month = commit.day[:7]
        if month != current_month:
            current_month = month
            lines.append(f"\n## {month}\n\n")
            lines.append("| Date | Change | Who | Why | Files | Lines |\n")
            lines.append("| --- | --- | --- | --- | --- | --- |\n")
        changed = len(commit.files)
        lines.append(
            "| {date} | `{short}` {subject} | {who} | {why} | {files} | +{add}/-{rem} |\n".format(
                date=commit.day,
                short=commit.short_hash,
                subject=_escape(commit.subject),
                who=_escape(commit.author),
                why=_escape(commit.why),
                files=changed,
                add=commit.insertions,
                rem=commit.deletions,
            )
        )

    lines.append("\n---\n\n## Files changed per commit\n\n")
    for commit in commits:
        lines.append(f"<details><summary><code>{commit.short_hash}</code> {_escape(commit.subject)}</summary>\n\n")
        for path in sorted(commit.files):
            lines.append(f"- `{_public_text(path)}`\n")
        if not commit.files:
            lines.append("- _no file changes recorded_\n")
        lines.append("\n</details>\n\n")
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when the ledger on disk differs from the history.",
    )
    parser.add_argument(
        "--output", type=Path, default=LEDGER_PATH, help="Where to write the ledger."
    )
    args = parser.parse_args()

    content = render(read_commits())

    if args.check:
        try:
            existing = args.output.read_text(encoding="utf-8")
        except OSError:
            print(f"{args.output} is missing. Run: python scripts/update_ledger.py", file=sys.stderr)
            return 1
        if existing != content:
            print(
                f"{args.output} is out of date. Run: python scripts/update_ledger.py",
                file=sys.stderr,
            )
            return 1
        print(f"{args.output} is up to date.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

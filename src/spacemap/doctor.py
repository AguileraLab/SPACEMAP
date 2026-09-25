"""``spacemap doctor``: check the environment SPACEMAP runs in, and say what to fix.

Run it as ``spacemap doctor`` (or ``python -m spacemap.doctor``). Each check
prints OK, NOTE or PROBLEM with one plain sentence, and a command or step to fix
it where there is one. The exit code is 1 when there is a PROBLEM.

What it checks:
- the Python version;
- which SPACEMAP code is loaded, and whether an older SPACEMAP install is
  still registered in the environment (it can shadow the current code);
- that every package SPACEMAP needs is installed, and that ``pip check`` finds
  no conflicts;
- the AI assistant's packages (optional);
- that only one OpenCV package is installed;
- the GPU;
- where SPACEMAP keeps its settings and models, and the free space there.
"""

from __future__ import annotations

import importlib.metadata as metadata
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

OK, NOTE, PROBLEM = "OK", "NOTE", "PROBLEM"
DISTRIBUTION = "spacemap-ai"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""


def _installed(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _space(gigabytes: float) -> str:
    return f"{gigabytes / 1000:.0f} TB" if gigabytes >= 1000 else f"{gigabytes:.0f} GB"


def check_python() -> Check:
    version = ".".join(map(str, sys.version_info[:3]))
    if sys.version_info < (3, 10):
        return Check("Python", PROBLEM, f"Python {version}; SPACEMAP needs 3.10 or newer.",
                     "Create the environment from environment.yml (Python 3.10).")
    if sys.version_info[:2] != (3, 10):
        return Check("Python", NOTE, f"Python {version}. The tested environment uses 3.10 (palom needs "
                     "scikit-image < 0.20, which has no builds for newer Python versions).")
    return Check("Python", OK, f"Python {version} at {sys.executable}")


def check_code_location() -> list[Check]:
    import spacemap

    code = Path(spacemap.__file__).resolve().parents[1]
    checks = [Check("SPACEMAP code", OK, f"Loaded from {code}")]
    if _installed(DISTRIBUTION) is None:
        checks.append(Check("SPACEMAP install", NOTE,
                            "SPACEMAP is not installed in this environment; it runs from the folder above.",
                            'From the repository folder: python -m pip install -e ".[ai]"'))
    old = _installed("spacemap")
    if old is not None:
        checks.append(Check(
            "Old SPACEMAP install", PROBLEM,
            f"An older SPACEMAP package ('spacemap' {old}) is still registered. Its requirements "
            "(for example deepcell) cause pip's 'requires ... which is not installed' errors, and its "
            "path file can make Python load old SPACEMAP code.",
            'python -m pip uninstall spacemap   (then: python -m pip install -e ".[ai]" in the repository)'))
    return checks


def check_requirements() -> Check:
    """Every requirement of the installed SPACEMAP (without extras) is installed at a fitting version."""
    try:
        from packaging.requirements import Requirement
    except ImportError:  # pragma: no cover - packaging comes with pip
        return Check("Required packages", NOTE, "Could not read requirements (the 'packaging' module is missing).")
    try:
        declared = metadata.requires(DISTRIBUTION) or []
    except metadata.PackageNotFoundError:
        return Check("Required packages", NOTE, "Skipped: SPACEMAP is not installed as a package.")
    missing, wrong = [], []
    for text in declared:
        requirement = Requirement(text)
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue  # an extra, or not for this platform
        version = _installed(requirement.name)
        if version is None:
            missing.append(requirement.name)
        elif requirement.specifier and not requirement.specifier.contains(version, prereleases=True):
            wrong.append(f"{requirement.name} {version} (needs {requirement.specifier})")
    if missing or wrong:
        detail = "; ".join(filter(None, [
            f"missing: {', '.join(missing)}" if missing else "",
            f"wrong version: {', '.join(wrong)}" if wrong else "",
        ]))
        return Check("Required packages", PROBLEM, detail,
                     'From the repository folder: python -m pip install -e ".[ai]"')
    return Check("Required packages", OK, f"All {len(declared)} requirements of SPACEMAP are met.")


def check_pip() -> Check:
    try:
        done = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Check("Package conflicts (pip check)", NOTE, f"Could not run pip check: {exc}")
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    if done.returncode == 0:
        return Check("Package conflicts (pip check)", OK, "No conflicts.")
    return Check("Package conflicts (pip check)", PROBLEM, " | ".join(lines[:8]) + (" ..." if len(lines) > 8 else ""),
                 "A clean environment from environment.yml has none; see docs/installation.md.")


def check_assistant() -> Check:
    from spacemap.ai_settings import AI_PACKAGES, dependency_report

    missing = [row["name"] for row in dependency_report(AI_PACKAGES) if row["required"] and not row["installed"]]
    if missing:
        return Check("AI assistant packages", NOTE, f"Not installed: {', '.join(missing)} (the assistant is optional).",
                     'python -m pip install -e ".[ai]"   (inside this same environment)')
    return Check("AI assistant packages", OK, "Installed.")


def check_opencv() -> Check:
    """palom 2024.x needs opencv-python, Cellpose and SPACEMAP the headless build: both, at one version."""
    versions = {name: _installed(name) for name in ("opencv-python", "opencv-python-headless")}
    present = {name: version for name, version in versions.items() if version}
    if not present:
        return Check("OpenCV", PROBLEM, "OpenCV is not installed.", 'python -m pip install -e ".[ai]"')
    if len(present) == 2 and len(set(present.values())) > 1:
        wanted = max(present.values())
        return Check("OpenCV", PROBLEM, "opencv-python and opencv-python-headless are at different versions "
                     f"({present['opencv-python']} and {present['opencv-python-headless']}); they share one "
                     "module, so the mix can break image reading.",
                     f"python -m pip install opencv-python=={wanted} opencv-python-headless=={wanted}")
    return Check("OpenCV", OK, ", ".join(f"{name} {version}" for name, version in present.items()))


def check_gpu() -> Check:
    try:
        import torch
    except ImportError:
        return Check("GPU", NOTE, "torch is not installed, so no GPU can be used.")
    if not torch.cuda.is_available():
        return Check("GPU", NOTE, "No GPU visible. SPACEMAP runs on the CPU; the AI assistant's larger models "
                     "need a GPU node.")
    props = torch.cuda.get_device_properties(0)
    return Check("GPU", OK, f"{props.name}, {props.total_memory / 1e9:.0f} GB (CUDA {torch.version.cuda})")


def check_folders() -> list[Check]:
    from spacemap import userpaths
    from spacemap.ai_settings import resolved_models_dir
    from spacemap.hardware import free_disk_gb

    home = userpaths.spacemap_home()
    checks = []
    if userpaths.is_temporary(home):
        checks.append(Check("Settings folder", PROBLEM, f"{home} is a temporary folder: it is cleaned and "
                            "differs between machines.", "Set SPACEMAP_HOME to a lasting folder."))
    else:
        checks.append(Check("Settings folder", OK, str(home)))
    models = Path(resolved_models_dir())
    free = free_disk_gb(models if models.exists() else models.parent)
    status = OK if not userpaths.is_temporary(models) else PROBLEM
    detail = f"{models} ({_space(free)} free)" if free else str(models)
    fix = ""
    if status == PROBLEM:
        detail += " is a temporary folder."
        fix = "Choose a models folder on lasting storage in the assistant's setup."
    elif free and free < 30:
        status, fix = NOTE, "Local models need 5-25 GB each; choose a larger models folder in the assistant's setup."
    checks.append(Check("Models folder", status, detail, fix))
    return checks


def run_checks(*, include_pip: bool = True) -> list[Check]:
    checks = [check_python(), *check_code_location(), check_requirements()]
    if include_pip:
        checks.append(check_pip())
    checks += [check_opencv(), check_assistant(), check_gpu(), *check_folders()]
    return checks


def format_report(checks: list[Check]) -> str:
    lines = ["SPACEMAP environment check", ""]
    for check in checks:
        lines.append(f"[{check.status:7s}] {check.name}: {check.detail}")
        if check.fix and check.status != OK:
            lines.append(f"          To fix: {check.fix}")
    problems = sum(check.status == PROBLEM for check in checks)
    lines += ["", "No problems found." if not problems else f"{problems} problem(s) to fix (see above)."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"-h", "--help"}:
        print(__doc__.strip())
        return 0
    checks = run_checks(include_pip="--no-pip" not in argv)
    print(format_report(checks))
    return 1 if any(check.status == PROBLEM for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())

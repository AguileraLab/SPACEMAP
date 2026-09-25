"""`spacemap doctor` reports problems in plain words and exits 1 when there is one."""

from __future__ import annotations

import subprocess

from spacemap import doctor


def _versions(monkeypatch, installed: dict):
    monkeypatch.setattr(doctor, "_installed", lambda name: installed.get(name))


def test_an_old_spacemap_install_is_a_problem(monkeypatch):
    _versions(monkeypatch, {"spacemap": "0.1.0", "spacemap-ai": "0.3.0"})
    checks = {c.name: c for c in doctor.check_code_location()}
    assert checks["Old SPACEMAP install"].status == doctor.PROBLEM
    assert "pip uninstall spacemap" in checks["Old SPACEMAP install"].fix
    _versions(monkeypatch, {"spacemap-ai": "0.3.0"})
    assert [c.status for c in doctor.check_code_location()] == [doctor.OK]


def test_two_opencv_packages_must_share_one_version(monkeypatch):
    # palom 2024.x needs opencv-python; Cellpose and SPACEMAP the headless build.
    _versions(monkeypatch, {"opencv-python": "4.11.0.86", "opencv-python-headless": "4.11.0.86"})
    assert doctor.check_opencv().status == doctor.OK
    _versions(monkeypatch, {"opencv-python": "4.10.0.84", "opencv-python-headless": "4.11.0.86"})
    check = doctor.check_opencv()
    assert check.status == doctor.PROBLEM and "opencv-python==4.11.0.86" in check.fix
    _versions(monkeypatch, {})
    assert doctor.check_opencv().status == doctor.PROBLEM


def test_pip_conflicts_are_a_problem(monkeypatch):
    def fake(*args, **kwargs):
        return subprocess.CompletedProcess(args, 1, "palom 2024.12.1 has requirement scikit-image<0.20\n", "")

    monkeypatch.setattr(doctor.subprocess, "run", fake)
    check = doctor.check_pip()
    assert check.status == doctor.PROBLEM and "scikit-image" in check.detail


def test_a_temporary_models_folder_is_a_problem(monkeypatch, tmp_path):
    import tempfile

    from spacemap import ai_settings

    monkeypatch.setattr(ai_settings, "resolved_models_dir", lambda: tempfile.gettempdir())
    models = next(c for c in doctor.check_folders() if c.name == "Models folder")
    assert models.status == doctor.PROBLEM


def test_the_report_and_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(doctor, "run_checks", lambda include_pip=True: [
        doctor.Check("Python", doctor.OK, "3.10"), doctor.Check("OpenCV", doctor.PROBLEM, "missing", "pip install")])
    assert doctor.main(["--no-pip"]) == 1
    out = capsys.readouterr().out
    assert "[PROBLEM] OpenCV: missing" in out and "To fix: pip install" in out and "1 problem(s)" in out
    monkeypatch.setattr(doctor, "run_checks", lambda include_pip=True: [doctor.Check("Python", doctor.OK, "3.10")])
    assert doctor.main([]) == 0


def test_the_launcher_runs_the_doctor(monkeypatch):
    from spacemap_gui import launch

    monkeypatch.setattr(doctor, "main", lambda argv: 7)
    assert launch.main(["doctor"]) == 7

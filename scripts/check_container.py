#!/usr/bin/env python3
"""Validate a built Docker image, its health check and persistent volumes."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
import uuid


def docker(*args: str, capture: bool = False) -> str:
    result = subprocess.run(["docker", *args], check=True, text=True,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else ""


def wait_healthy(container: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = json.loads(docker("inspect", "--format", "{{json .State}}", container, capture=True))
        if not state.get("Running"):
            raise RuntimeError("Container exited during startup")
        health = state.get("Health", {}).get("Status")
        if health == "healthy":
            return
        if health == "unhealthy":
            raise RuntimeError("The image health check reported unhealthy")
        time.sleep(2)
    raise TimeoutError("Container did not become healthy within the startup timeout")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="spacemap:validation")
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    user = docker("image", "inspect", "--format", "{{.Config.User}}", args.image, capture=True)
    if not user or user.split(":")[0] in ("root", "0"):
        raise RuntimeError("The image must run as a non-root user")

    docker("run", "--rm", "--network=none", args.image, "python", "-m", "pip", "check")
    docker("run", "--rm", "--network=none", args.image, "python",
           "/opt/spacemap/scripts/check_runtime.py", "--device", "cpu")
    docker("run", "--rm", "--network=none", args.image, "python",
           "/opt/spacemap/scripts/check_installed_app.py")

    token = "spacemap-check-" + uuid.uuid4().hex[:12]
    container = token + "-app"
    volumes = [token + "-projects", token + "-state"]
    created_volumes = []
    paths = ["/workspace/projects", "/home/spacemap/.spacemap"]
    mounts = [arg for volume, path in zip(volumes, paths) for arg in ("--mount", f"type=volume,src={volume},dst={path}")]
    try:
        for volume in volumes:
            docker("volume", "create", volume)
            created_volumes.append(volume)
        docker("run", "-d", "--name", container, "-p", "127.0.0.1::8501", "--shm-size=2g",
               "-e", "SPACEMAP_ASSISTANT=off", *mounts, args.image)
        wait_healthy(container, args.timeout)
        address = docker("port", container, "8501/tcp", capture=True).splitlines()[0]
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://{address}/_stcore/health", timeout=10) as response:
            assert response.status == 200 and response.read().strip() == b"ok"
        docker("exec", container, "python", "-c",
               "import os; from pathlib import Path; assert os.getuid() != 0; "
               f"[(Path(p) / 'validation.txt').write_text({token!r}) for p in {paths!r}]")
        docker("rm", "-f", container)
        docker("run", "--rm", "--network=none", *mounts, args.image, "python", "-c",
               "from pathlib import Path; "
               f"assert all((Path(p) / 'validation.txt').read_text() == {token!r} for p in {paths!r})")
        print("PASS: non-root image, installed GUI, CPU operators, HTTP health and persistent volume writes")
    except Exception:
        subprocess.run(["docker", "logs", "--tail", "100", container], check=False)
        raise
    finally:
        subprocess.run(["docker", "rm", "-f", container], check=False, capture_output=True)
        for volume in created_volumes:
            subprocess.run(["docker", "volume", "rm", volume], check=False)


if __name__ == "__main__":
    main()

"""Screenshot classic SPACEMAP (Streamlit) in headless Firefox, after the page has drawn, and measure elements.

Streamlit draws after the page loads, so ``firefox --screenshot`` catches a blank page. This drives
Firefox over its built-in remote protocol (WebDriver BiDi): open the app, wait until a selector
appears, report the boxes of the elements asked for, and save a PNG. It starts its own Streamlit on
a free port with a throwaway SPACEMAP home, so it never touches the user's settings or projects.

Usage: PYTHONPATH=src:tests python tests/classic_page_screenshot.py OUT.png [--wait CSS] [--measure CSS ...]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_port(port: int, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.5)
    raise TimeoutError(f"Nothing answered on port {port}.")


async def drive(url: str, bidi_port: int, wait: str, measure: list[str], out: Path, width: int, height: int) -> dict:
    import websockets

    async with websockets.connect(f"ws://127.0.0.1:{bidi_port}/session", max_size=64 * 1024 * 1024, proxy=None) as ws:
        counter = 0

        async def call(method, params):
            nonlocal counter
            counter += 1
            await ws.send(json.dumps({"id": counter, "method": method, "params": params}))
            while True:
                message = json.loads(await ws.recv())
                if message.get("id") == counter:
                    if message.get("type") == "error":
                        raise RuntimeError(f"{method}: {message.get('message')}")
                    return message["result"]

        await call("session.new", {"capabilities": {}})
        context = (await call("browsingContext.getTree", {}))["contexts"][0]["context"]
        await call("browsingContext.setViewport", {"context": context, "viewport": {"width": width, "height": height}})
        await call("browsingContext.navigate", {"context": context, "url": url, "wait": "complete"})
        script = (
            "(() => { const out = {ready: !!document.querySelector(%s), boxes: {}};"
            " for (const css of %s) out.boxes[css] = Array.from(document.querySelectorAll(css)).map(e => {"
            " const b = e.getBoundingClientRect(); return {top: b.top, bottom: b.bottom, left: b.left, height: b.height,"
            " text: (e.innerText || '').trim().slice(0, 40)}; }); return JSON.stringify(out); })()"
        ) % (json.dumps(wait), json.dumps(measure))
        found = {}
        for _ in range(240):
            result = await call("script.evaluate", {"expression": script, "target": {"context": context}, "awaitPromise": False})
            found = json.loads(result["result"]["value"])
            if found["ready"]:
                break
            await asyncio.sleep(0.5)
        await asyncio.sleep(2)  # fonts and icons settle
        result = await call("script.evaluate", {"expression": script, "target": {"context": context}, "awaitPromise": False})
        found = json.loads(result["result"]["value"])
        shot = await call("browsingContext.captureScreenshot", {"context": context})
        out.write_bytes(base64.b64decode(shot["data"]))
        return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--wait", default='[data-testid="stPopover"]')
    parser.add_argument("--measure", nargs="*", default=[])
    parser.add_argument("--size", default="1680,500")
    args = parser.parse_args()
    width, height = (int(v) for v in args.size.split(","))
    with tempfile.TemporaryDirectory(prefix="spacemap-classic-shot-") as folder:
        # A throwaway home as well: classic keeps its project list in ~/.spacemap_gui, so no real project opens.
        env = {**os.environ, "SPACEMAP_HOME": str(Path(folder) / "home"), "HOME": str(Path(folder) / "user"),
               "SPACEMAP_ASSISTANT": "on", "MPLCONFIGDIR": folder}
        (Path(folder) / "user").mkdir()
        app_port, bidi_port = free_port(), free_port()
        app = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve().parents[1] / "src/spacemap_gui/app.py"),
             "--server.headless", "true", "--server.port", str(app_port), "--server.address", "127.0.0.1",
             "--browser.gatherUsageStats", "false"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        profile = Path(folder) / "firefox"
        profile.mkdir()
        (profile / "user.js").write_text('user_pref("network.proxy.type", 0);\n')
        browser = subprocess.Popen(
            ["firefox", "--headless", "--no-remote", "--profile", str(profile), f"--remote-debugging-port={bidi_port}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        try:
            wait_for_port(app_port, 120)
            wait_for_port(bidi_port, 60)
            found = asyncio.run(drive(f"http://127.0.0.1:{app_port}/", bidi_port, args.wait, args.measure, args.out, width, height))
            print(json.dumps(found, indent=1))
        finally:
            for process in (browser, app):
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == "__main__":
    main()

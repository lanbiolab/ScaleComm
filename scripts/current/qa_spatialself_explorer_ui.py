#!/usr/bin/env python
"""Regenerate and QA the SpatialSelf Explorer UI.

The script produces reproducible screenshots for the full tool view and the
paper screenshot view, then runs a small Chrome DevTools Protocol smoke test
for the core interactions.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs/current_web_ui"
PRODUCT_HTML = OUT_DIR / "spatialself_explorer_product_ui.html"
PAPER_HTML = OUT_DIR / "spatialself_explorer_paper_screenshot.html"
TOOL_PNG = OUT_DIR / "spatialself_explorer_product_ui_qc_tool.png"
PAPER_PNG = OUT_DIR / "spatialself_explorer_product_ui_qc_paper.png"
MANIFEST = OUT_DIR / "spatialself_explorer_ui_qc_manifest.json"


def run(cmd: list[str], *, cwd: Path = ROOT, timeout: int = 120) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, timeout=timeout)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def chrome_base_args() -> list[str]:
    return [
        "google-chrome",
        "--headless=new",
        "--no-sandbox",
        "--enable-unsafe-swiftshader",
        "--use-gl=swiftshader",
        "--ignore-gpu-blocklist",
        "--allow-file-access-from-files",
    ]


def capture_screenshot(url: str, out_png: Path, width: int = 1800, height: int = 1000) -> None:
    cmd = chrome_base_args() + [
        f"--window-size={width},{height}",
        "--force-device-scale-factor=2",
        "--virtual-time-budget=22000",
        f"--screenshot={out_png}",
        url,
    ]
    run(cmd, timeout=90)


def wait_for_cdp(port: int, timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    url = f"http://127.0.0.1:{port}/json"
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as handle:
                pages = json.load(handle)
            for page in pages:
                if page.get("type") == "page":
                    return page
        except Exception as exc:  # pragma: no cover - diagnostics only
            last_err = exc
            time.sleep(0.5)
    raise RuntimeError(f"Chrome DevTools did not become ready: {last_err}")


def cdp_smoke(url: str) -> dict:
    try:
        import websocket
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("Missing websocket-client package for CDP smoke test") from exc

    port = find_free_port()
    with tempfile.TemporaryDirectory(prefix="spatialself_chrome_", ignore_cleanup_errors=True) as user_data:
        proc = subprocess.Popen(
            chrome_base_args()
            + [
                "--window-size=1800,1000",
                f"--remote-debugging-port={port}",
                "--remote-allow-origins=*",
                f"--user-data-dir={user_data}",
                url,
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            page = wait_for_cdp(port)
            ws = websocket.create_connection(
                page["webSocketDebuggerUrl"],
                timeout=10,
                origin=f"http://127.0.0.1:{port}",
            )
            seq = 0

            def call(method: str, params: dict | None = None) -> dict:
                nonlocal seq
                seq += 1
                ws.send(json.dumps({"id": seq, "method": method, "params": params or {}}))
                while True:
                    msg = json.loads(ws.recv())
                    if msg.get("id") == seq:
                        return msg

            call("Runtime.enable")
            call(
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": 1800,
                    "height": 1000,
                    "deviceScaleFactor": 2,
                    "mobile": False,
                },
            )
            time.sleep(5)

            def eval_js(expr: str):
                res = call(
                    "Runtime.evaluate",
                    {"expression": expr, "returnByValue": True, "awaitPromise": True},
                )
                if "exceptionDetails" in res:
                    raise RuntimeError(res["exceptionDetails"])
                return res["result"]["result"].get("value")

            def wait_js(expr: str, expected=True, timeout: float = 18.0) -> None:
                deadline = time.time() + timeout
                last = None
                while time.time() < deadline:
                    last = eval_js(expr)
                    if last == expected:
                        return
                    time.sleep(0.5)
                raise RuntimeError(f"Timed out waiting for {expr!r}; last value={last!r}")

            wait_js(
                "Boolean(document.querySelector('#plot .main-svg') || document.querySelector('#plot canvas'))"
            )
            time.sleep(2.5)

            result = {
                "initial_lr": eval_js("document.querySelector('#ins-lr').textContent"),
                "initial_visible_edges": eval_js("edgeIndicesForPair(true).length.toLocaleString()"),
                "initial_filter_pool": eval_js("edgeIndicesForPair(false).length.toLocaleString()"),
                "initial_page_class": eval_js("document.querySelector('.page').className"),
                "background_unannotated": eval_js("DATA.background.cell_type.filter(x => x === 'unannotated').length"),
                "initial_unannotated_edges": eval_js(
                    "DATA.lrs[document.querySelector('#ins-lr').textContent].edges.sender_type"
                    ".filter((x,i) => x === 'unannotated' || "
                    "DATA.lrs[document.querySelector('#ins-lr').textContent].edges.receiver_type[i] === 'unannotated').length"
                ),
            }
            initial_capture = call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
            initial_png = OUT_DIR / "spatialself_explorer_product_ui_qc_cdp_initial.png"
            initial_png.write_bytes(base64.b64decode(initial_capture["result"]["data"]))
            result["cdp_initial_screenshot"] = str(initial_png)
            eval_js("document.querySelector('#paper-view').click()")
            time.sleep(3)
            result["paper_page_class"] = eval_js("document.querySelector('.page').className")
            paper_capture = call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
            paper_png = OUT_DIR / "spatialself_explorer_product_ui_qc_cdp_paper.png"
            paper_png.write_bytes(base64.b64decode(paper_capture["result"]["data"]))
            result["cdp_paper_screenshot"] = str(paper_png)
            eval_js("document.querySelector('#tool-view').click()")
            time.sleep(1)
            result["tool_page_class_after_toggle"] = eval_js("document.querySelector('.page').className")
            eval_js(
                "document.querySelector('#lr-search').value='Reln'; "
                "document.querySelector('#lr-search').dispatchEvent(new Event('input',{bubbles:true}));"
            )
            time.sleep(1)
            result["filtered_lr_options"] = eval_js("document.querySelectorAll('#lr-select option').length")
            eval_js(
                "document.querySelector('#lr-select').value='Reln-Lrp8'; "
                "document.querySelector('#lr-select').dispatchEvent(new Event('change',{bubbles:true}));"
            )
            time.sleep(3)
            result["reln_lr"] = eval_js("document.querySelector('#ins-lr').textContent")
            result["reln_visible_edges"] = eval_js("edgeIndicesForPair(true).length.toLocaleString()")
            eval_js(
                "document.querySelector('#pair-select').value='All interactions'; "
                "document.querySelector('#pair-select').dispatchEvent(new Event('change',{bubbles:true}));"
            )
            time.sleep(3)
            result["reln_all_visible_edges"] = eval_js("edgeIndicesForPair(true).length.toLocaleString()")

            # Capture a small sanity screenshot from the same browser session so
            # failures are easier to debug when the CLI screenshot path changes.
            capture = call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
            debug_png = OUT_DIR / "spatialself_explorer_product_ui_qc_cdp.png"
            debug_png.write_bytes(base64.b64decode(capture["result"]["data"]))
            result["cdp_debug_screenshot"] = str(debug_png)
            ws.close()
            return result
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def assert_smoke(result: dict) -> None:
    expected = {
        "initial_lr": "Cxcl12-Cxcr4",
        "initial_visible_edges": "128",
        "initial_filter_pool": "638",
        "background_unannotated": 0,
        "initial_unannotated_edges": 0,
        "filtered_lr_options": 2,
        "reln_lr": "Reln-Lrp8",
        "reln_visible_edges": "77",
        "reln_all_visible_edges": "77",
    }
    failures = []
    for key, value in expected.items():
        if result.get(key) != value:
            failures.append(f"{key}: expected {value!r}, observed {result.get(key)!r}")
    if "paper-view" not in str(result.get("paper_page_class", "")):
        failures.append(f"paper mode did not activate: {result.get('paper_page_class')!r}")
    restored_class = str(result.get("tool_page_class_after_toggle"))
    if "paper-view" in restored_class or "presentation-view" in restored_class:
        failures.append(f"tool mode did not restore: {result.get('tool_page_class_after_toggle')!r}")
    if failures:
        raise AssertionError("\n".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true", help="Use existing HTML files.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.skip_build:
        run([sys.executable, "scripts/current/build_spatialself_explorer_product_ui.py"], timeout=120)

    smoke = cdp_smoke(PRODUCT_HTML.resolve().as_uri())
    assert_smoke(smoke)
    # The product screenshot is captured from the CDP session after Plotly has
    # rendered; this is more reliable than Chrome's one-shot --screenshot for
    # large WebGL/canvas payloads.
    cdp_initial = Path(smoke["cdp_initial_screenshot"])
    cdp_initial.replace(TOOL_PNG)
    cdp_paper = Path(smoke["cdp_paper_screenshot"])
    cdp_paper.replace(PAPER_PNG)

    manifest = {
        "product_html": str(PRODUCT_HTML),
        "paper_html": str(PAPER_HTML),
        "tool_screenshot": str(TOOL_PNG),
        "paper_screenshot": str(PAPER_PNG),
        "smoke_test": smoke,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

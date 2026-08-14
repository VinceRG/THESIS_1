"""
Capture full-page screenshots for every GET route registered by the Flask app.

Usage:
    python scripts/screenshot_routes.py

Optional environment variables:
    SCREENSHOT_BASE_URL=http://127.0.0.1:5000
    SCREENSHOT_USERNAME=admin
    SCREENSHOT_PASSWORD=admin123
    SCREENSHOT_TIMEOUT_MS=30000
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from werkzeug.routing import BuildError

ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT_DIR = ROOT / "screenshots"
BASE_URL = os.environ.get("SCREENSHOT_BASE_URL", "http://127.0.0.1:5000").rstrip("/")
USERNAME = os.environ.get("SCREENSHOT_USERNAME", "admin")
PASSWORD = os.environ.get("SCREENSHOT_PASSWORD", "admin123")
TIMEOUT_MS = int(os.environ.get("SCREENSHOT_TIMEOUT_MS", "30000"))

SAMPLE_VALUES = {
    "int": {
        "appointment_id": 1,
        "branch_id": 1,
        "package_id": 1,
        "patient_id": 1,
        "service_id": 1,
        "staff_id": 1,
        "user_id": 1,
    },
    "string": {
        "report_key": "monthly-consultation",
    },
}


def is_server_running() -> bool:
    try:
        with urlopen(f"{BASE_URL}/login", timeout=3) as response:
            return response.status < 500
    except URLError:
        return False
    except Exception:
        return False


def start_flask_server() -> subprocess.Popen[str] | None:
    if is_server_running():
        return None

    env = os.environ.copy()
    env.setdefault("FLASK_ENV", "development")
    process = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    deadline = time.time() + 45
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Flask server stopped before it became available.")
        if is_server_running():
            return process
        time.sleep(1)

    process.terminate()
    raise RuntimeError(f"Flask server did not respond at {BASE_URL} within 45 seconds.")


def sample_arguments(rule: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for argument in rule.arguments:
        converter = rule._converters[argument].__class__.__name__.lower()
        if "integer" in converter or converter == "intconverter":
            values[argument] = SAMPLE_VALUES["int"].get(argument, 1)
        else:
            values[argument] = SAMPLE_VALUES["string"].get(argument, f"sample-{argument}")
    return values


def discover_routes() -> list[dict[str, str]]:
    sys.path.insert(0, str(ROOT))
    from app import app  # noqa: PLC0415

    routes: list[dict[str, str]] = []
    with app.test_request_context():
        for rule in sorted(app.url_map.iter_rules(), key=lambda item: item.rule):
            if rule.endpoint == "static" or "GET" not in rule.methods:
                continue
            try:
                path = app.url_for(rule.endpoint, **sample_arguments(rule))
            except BuildError as exc:
                routes.append(
                    {
                        "endpoint": rule.endpoint,
                        "rule": rule.rule,
                        "path": "",
                        "error": f"could not build URL: {exc}",
                    }
                )
                continue
            routes.append({"endpoint": rule.endpoint, "rule": rule.rule, "path": path})
    return routes


def safe_filename(route: dict[str, str]) -> str:
    source = f"{route['endpoint']} {route.get('path') or route['rule']}".strip()
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", source).strip("._-").lower()
    return f"{filename[:140] or 'route'}.png"


def login(page: Any) -> None:
    page.goto(f"{BASE_URL}/login", wait_until="networkidle", timeout=TIMEOUT_MS)
    page.fill("input[name='username']", USERNAME)
    page.fill("input[name='password']", PASSWORD)
    page.click("button[type='submit']")
    page.wait_for_load_state("networkidle", timeout=TIMEOUT_MS)


def capture_route(page: Any, route: dict[str, str], timeout_error: type[Exception]) -> dict[str, Any]:
    if not route.get("path"):
        return {**route, "status": "skipped", "screenshot": None}

    url = f"{BASE_URL}{route['path']}"
    output_path = SCREENSHOT_DIR / safe_filename(route)
    result: dict[str, Any] = {
        "endpoint": route["endpoint"],
        "rule": route["rule"],
        "path": route["path"],
        "url": url,
        "screenshot": str(output_path.relative_to(ROOT)),
    }

    try:
        response = page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
        try:
            page.wait_for_load_state("networkidle", timeout=TIMEOUT_MS)
        except timeout_error:
            result["load_warning"] = "networkidle timeout; captured after DOMContentLoaded"
        result["http_status"] = response.status if response else None
        page.screenshot(path=output_path, full_page=True)
        result["status"] = "captured"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)

    return result


def main() -> int:
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError as exc:
        print("Playwright is not installed. Run: python -m pip install playwright")
        print("Then install a browser with: python -m playwright install chromium")
        raise SystemExit(1) from exc

    SCREENSHOT_DIR.mkdir(exist_ok=True)
    server_process = start_flask_server()
    routes = discover_routes()
    results: list[dict[str, Any]] = []

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            try:
                login(page)
            except Exception as exc:
                results.append({"endpoint": "login", "status": "error", "error": str(exc)})

            for route in routes:
                results.append(capture_route(page, route, PlaywrightTimeoutError))

            browser.close()
    finally:
        if server_process is not None:
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_process.kill()

    report_path = SCREENSHOT_DIR / "screenshot_report.json"
    report_path.write_text(
        json.dumps(
            {
                "base_url": BASE_URL,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "routes": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    captured = sum(1 for item in results if item.get("status") == "captured")
    errored = sum(1 for item in results if item.get("status") == "error")
    skipped = sum(1 for item in results if item.get("status") == "skipped")
    print(f"Captured {captured} route screenshots. Errors: {errored}. Skipped: {skipped}.")
    print(f"Report: {report_path}")
    return 0 if errored == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Web dashboard entry point -- an alternative to main.py's CLI.

Usage:
    python webapp.py

Starts the bridge plus a local dashboard where you browse previously-seen
Figma files (a gallery built automatically from files you've opened with the
plugin running) and submit instructions from the browser instead of the CLI.

Unlike main.py, this boots fine with no model credentials: you can enter them
in the dashboard's Settings panel instead of editing .env.
"""
from __future__ import annotations

import time
import webbrowser

from bridge.server import Bridge
from web.app import DashboardServer, serve
from web.registry import Registry
from web.settings_store import SettingsStore

DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8787


def main() -> int:
    settings_store = SettingsStore()
    settings = settings_store.effective()

    bridge = Bridge(settings.bridge_host, settings.bridge_port)
    bridge.start()

    dashboard = DashboardServer(bridge, settings_store, Registry())
    serve(dashboard, DASHBOARD_HOST, DASHBOARD_PORT)

    url = f"http://{DASHBOARD_HOST}:{DASHBOARD_PORT}"
    print(f"  Bridge     ws://{settings.bridge_host}:{settings.bridge_port}")
    print(f"  Dashboard  {url}")
    if not settings.is_model_configured:
        print("  Model      not configured -- add your API details in Settings")
    else:
        print(f"  Model      {settings.model_name}")
    print("\nPress Ctrl+C to stop.")

    try:
        webbrowser.open(url)
    except Exception:
        pass  # headless or no browser -- the URL is printed above

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        bridge.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

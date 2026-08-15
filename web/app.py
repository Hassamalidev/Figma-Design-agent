"""Local HTTP server for the web dashboard: serves the single-page UI and a
small JSON API that lets it list known Figma files, kick off a run, and poll
progress.

Same "few and boring" dependency policy as the rest of this project --
stdlib `http.server` only, no web framework. All agent logic still lives in
agent/loop.py; this module just wires a browser UI to it.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import uuid

from agent import loop
from agent.llm import ModelClient, build_critic_client
from bridge.server import Bridge
from tools.figma_read import get_screenshot
from web.history import History, HistoryEntry
from web.registry import FileEntry, Registry
from web.settings_store import SettingsStore, mask

STATIC_DIR = Path(__file__).parent / "static"
CONNECT_POLL_INTERVAL = 1.0
MAX_WAIT_FOR_FILE_SECONDS = 600  # give up waiting for the target file after 10 minutes


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _RunLogHandler(logging.Handler):
    """Feeds agent.loop's log records into the run state the UI polls."""

    def __init__(self, sink: list[str], lock: threading.Lock):
        super().__init__()
        self._sink = sink
        self._lock = lock

    def emit(self, record: logging.LogRecord) -> None:
        with self._lock:
            self._sink.append(self.format(record))


class DashboardServer:
    """Owns run + registry state; the HTTP handler just reads/writes through it."""

    def __init__(
        self,
        bridge: Bridge,
        settings_store: SettingsStore,
        registry: Registry,
        llm_factory=ModelClient,
        history: History | None = None,
    ):
        self.bridge = bridge
        self.settings_store = settings_store
        self.registry = registry
        self.history = history or History()
        # Injected so credentials entered in the UI take effect immediately --
        # the client is rebuilt per run rather than captured once at startup.
        self._llm_factory = llm_factory
        self._lock = threading.Lock()
        self._run_status = "idle"  # idle | waiting_for_file | running | done | error
        self._run_log: list[str] = []
        self._run_result: dict | None = None
        self._last_captured_generation = -1

    # -- connection watcher: builds the gallery over time, unattended -------

    def watch_connections(self) -> None:
        """Whenever a new plugin connection reports a file, snapshot it into
        the registry. Runs for the process's lifetime on its own thread.
        """
        while True:
            generation = self.bridge.connection_generation
            identity = self.bridge.current_file
            if generation != self._last_captured_generation and identity and identity.file_key:
                self._capture(identity.file_key, identity.file_name)
                self._last_captured_generation = generation
            time.sleep(CONNECT_POLL_INTERVAL)

    def _capture(self, file_key: str, file_name: str) -> None:
        shot = get_screenshot(self.bridge)
        thumbnail = shot["image_base64"] if shot["ok"] else None
        self.registry.upsert(
            FileEntry(file_key=file_key, file_name=file_name, thumbnail_base64=thumbnail, last_seen=_now())
        )

    # -- run orchestration ---------------------------------------------------

    def start_run(self, file_key: str, instruction: str) -> tuple[bool, str]:
        if not self.settings_store.effective().is_model_configured:
            return False, "No model configured. Open Settings and add your API details first."
        with self._lock:
            if self._run_status in ("waiting_for_file", "running"):
                return False, "A run is already in progress."
            entry = self.registry.get(file_key)
            if entry is None:
                return False, f"Unknown file: {file_key}"
            self._run_status = "waiting_for_file"
            self._run_log = [
                f"Waiting for '{entry.file_name}' to connect -- open it in Figma Desktop "
                "and run the plugin if it isn't already."
            ]
            self._run_result = None
        threading.Thread(
            target=self._run_worker, args=(file_key, entry.file_name, instruction), daemon=True
        ).start()
        return True, "started"

    def _run_worker(self, file_key: str, file_name: str, instruction: str) -> None:
        started_at = _now()
        if not self._wait_for_file(file_key):
            with self._lock:
                self._run_status = "error"
                self._run_log.append(f"Timed out waiting for '{file_name}' to connect.")
            self._record_history(file_key, file_name, instruction, started_at, None)
            return

        with self._lock:
            self._run_status = "running"
            self._run_log.append(f"Connected. Running: {instruction}")

        # "agent" (not "agent.loop") so agent.planner's brief/plan logs -- and
        # anything else under agent.* -- flow through too, via propagation.
        logger = logging.getLogger("agent")
        handler = _RunLogHandler(self._run_log, self._lock)
        handler.setFormatter(logging.Formatter("%(message)s"))
        previous_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            settings = self.settings_store.effective()  # pick up credentials saved since startup
            prefs = self.settings_store.prefs()
            llm = self._llm_factory(
                settings.model_base_url, settings.model_api_key, settings.model_name
            )
            result = loop.run(
                instruction,
                self.bridge,
                llm,
                int(prefs["max_retries"]),
                int(prefs["max_steps"]),
                visual_gate=bool(prefs["visual_gate"]),
                critic_llm=build_critic_client(settings),
            )
            # Best-effort: the design is already built, so a failed thumbnail
            # refresh must not report the whole run as crashed and throw the
            # result away.
            try:
                self._capture(file_key, file_name)
            except Exception as exc:
                logger.info("Could not refresh the file thumbnail: %s", exc)
            with self._lock:
                self._run_status = "done"
                self._run_result = {
                    "success": result.success,
                    "created_node_count": len(result.created_node_ids),
                    "failed_steps": result.failed_steps,
                    "warnings": result.warnings,
                    "layout_defects": result.layout_defects,
                    "final_screenshot_base64": result.final_screenshot_base64,
                }
            self._record_history(file_key, file_name, instruction, started_at, result)
        except Exception as exc:  # the UI must hear about this, not spin forever
            with self._lock:
                self._run_status = "error"
                self._run_log.append(f"Run crashed: {exc}")
            self._record_history(file_key, file_name, instruction, started_at, None)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)

    def _record_history(self, file_key, file_name, instruction, started_at, result) -> None:
        """A run that failed is still worth remembering -- that's the point of a log."""
        try:
            self.history.add(
                HistoryEntry(
                    id=uuid.uuid4().hex[:12],
                    instruction=instruction,
                    file_key=file_key,
                    file_name=file_name,
                    status="done" if result is not None else "error",
                    success=bool(result.success) if result is not None else False,
                    created_node_count=len(result.created_node_ids) if result is not None else 0,
                    failed_step_count=len(result.failed_steps) if result is not None else 0,
                    started_at=started_at,
                    finished_at=_now(),
                    thumbnail_base64=(result.final_screenshot_base64 if result is not None else None),
                )
            )
        except Exception as exc:  # history must never break a run
            logging.getLogger(__name__).info("Could not record history: %s", exc)

    def _wait_for_file(self, file_key: str) -> bool:
        deadline = time.monotonic() + MAX_WAIT_FOR_FILE_SECONDS
        while time.monotonic() < deadline:
            identity = self.bridge.current_file
            if identity is not None and identity.file_key == file_key:
                return True
            time.sleep(CONNECT_POLL_INTERVAL)
        return False

    # -- read-only snapshots for the API --------------------------------------

    def files_snapshot(self) -> list[dict]:
        connected_key = self.bridge.current_file.file_key if self.bridge.current_file else None
        return [
            {
                "file_key": e.file_key,
                "file_name": e.file_name,
                "thumbnail_base64": e.thumbnail_base64,
                "last_seen": e.last_seen,
                "connected": e.file_key == connected_key,
            }
            for e in self.registry.list_files()
        ]

    def status_snapshot(self) -> dict:
        with self._lock:
            identity = self.bridge.current_file
            return {
                "status": self._run_status,
                "log": list(self._run_log),
                "result": self._run_result,
                "connected_file": (
                    {"file_key": identity.file_key, "file_name": identity.file_name} if identity else None
                ),
                "plugin_connected": self.bridge.is_connected,
                "model_configured": self.settings_store.effective().is_model_configured,
            }

    # -- settings + setup ------------------------------------------------------

    def settings_snapshot(self) -> dict:
        view = self.settings_store.view()
        s = view.settings
        return {
            "model_base_url": s.model_base_url,
            "model_name": s.model_name,
            "model_api_key_masked": mask(s.model_api_key),  # never send the real key
            "has_api_key": bool(s.model_api_key),
            "sources": view.sources,
            "configured": s.is_model_configured,
            "bridge_url": f"ws://{s.bridge_host}:{s.bridge_port}",
            "prefs": self.settings_store.prefs(),
        }

    def history_snapshot(self) -> list[dict]:
        return [e.summary() for e in self.history.list_entries()]

    def update_prefs(self, values: dict) -> dict:
        self.settings_store.update_prefs(values)
        return self.settings_snapshot()

    def update_settings(self, values: dict) -> dict:
        # An ABSENT field means "leave it alone"; an explicit empty string
        # means "clear it". Without that distinction a partial update (the UI
        # omits the API key when the user leaves it blank) would silently wipe
        # the fields it didn't mention.
        editable = ("model_base_url", "model_api_key", "model_name")
        self.settings_store.update({k: values[k] for k in editable if k in values})
        return self.settings_snapshot()

    def test_model_connection(self) -> dict:
        """Make one real, tiny call so the user finds out here rather than
        halfway through a run."""
        settings = self.settings_store.effective()
        if not settings.is_model_configured:
            return {"ok": False, "error": "Fill in the base URL, API key and model name first."}
        try:
            llm = self._llm_factory(
                settings.model_base_url, settings.model_api_key, settings.model_name
            )
            message = llm.complete(
                [{"role": "user", "content": "Reply with the single word: ok"}], tools=None
            )
            return {"ok": True, "reply": (message.content or "").strip()[:120]}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def setup_snapshot(self) -> dict:
        s = self.settings_store.effective()
        project_root = Path(__file__).resolve().parent.parent
        return {
            "manifest_path": str(project_root / "figma_plugin" / "manifest.json"),
            "plugin_dir": str(project_root / "figma_plugin"),
            "bridge_url": f"ws://{s.bridge_host}:{s.bridge_port}",
            "plugin_connected": self.bridge.is_connected,
            "connected_file": (
                self.bridge.current_file.file_name if self.bridge.current_file else None
            ),
        }


def _make_handler(dashboard: DashboardServer) -> type[BaseHTTPRequestHandler]:
    """BaseHTTPRequestHandler is instantiated per-request by http.server, so
    dashboard state has to reach it via this closure."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format_: str, *args) -> None:
            pass  # quiet -- the dashboard's own log panel is what matters

        def _send_json(self, status: int, payload) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_static(self, name: str, content_type: str) -> None:
            body = (STATIC_DIR / name).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            routes = {
                "/api/files": dashboard.files_snapshot,
                "/api/status": dashboard.status_snapshot,
                "/api/settings": dashboard.settings_snapshot,
                "/api/setup": dashboard.setup_snapshot,
                "/api/history": dashboard.history_snapshot,
            }
            if self.path in ("/", "/index.html"):
                self._send_static("index.html", "text/html")
            elif self.path in routes:
                self._send_json(200, routes[self.path]())
            else:
                self._send_json(404, {"error": "not found"})

        def _read_json(self) -> dict | None:
            length = int(self.headers.get("Content-Length", 0))
            try:
                return json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid JSON"})
                return None

        def do_POST(self) -> None:
            if self.path == "/api/run":
                self._handle_run()
            elif self.path == "/api/settings":
                self._handle_settings()
            elif self.path == "/api/settings/test":
                result = dashboard.test_model_connection()
                self._send_json(200 if result["ok"] else 400, result)
            elif self.path == "/api/prefs":
                payload = self._read_json()
                if payload is not None:
                    self._send_json(200, dashboard.update_prefs(payload))
            elif self.path == "/api/history/clear":
                dashboard.history.clear()
                self._send_json(200, {"ok": True})
            else:
                self._send_json(404, {"error": "not found"})

        def _handle_run(self) -> None:
            payload = self._read_json()
            if payload is None:
                return
            file_key = payload.get("file_key")
            instruction = (payload.get("instruction") or "").strip()
            if not file_key or not instruction:
                self._send_json(400, {"error": "file_key and instruction are required"})
                return
            ok, message = dashboard.start_run(file_key, instruction)
            self._send_json(202 if ok else 409, {"ok": ok, "message": message})

        def _handle_settings(self) -> None:
            payload = self._read_json()
            if payload is None:
                return
            self._send_json(200, dashboard.update_settings(payload))

    return Handler


def serve(dashboard: DashboardServer, host: str, port: int) -> ThreadingHTTPServer:
    """Start the dashboard's HTTP server and connection watcher on background threads."""
    server = ThreadingHTTPServer((host, port), _make_handler(dashboard))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    threading.Thread(target=dashboard.watch_connections, daemon=True).start()
    return server

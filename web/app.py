"""Local HTTP server for the web dashboard: serves the single-page UI and a
small JSON API that lets it list known Figma files, kick off a run, and poll
progress.

Same "few and boring" dependency policy as the rest of this project --
stdlib `http.server` only, no web framework. All agent logic still lives in
agent/loop.py; this module just wires a browser UI to it.
"""
from __future__ import annotations

import base64
import json
import logging
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import uuid

from agent import edit_loop, loop, reference, scaffold
from agent.llm import ModelClient, build_critic_client, build_vision_client
from agent.metrics import RunMetrics
from bridge.server import Bridge
from tools.figma_exec import execute_figma_js
from tools.figma_read import get_screenshot
from web.history import History, HistoryEntry
from web.registry import FileEntry, Registry
from web.settings_store import EDITABLE, SettingsStore, mask

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
        self._run_mode = "create"  # create | edit -- declared by the user, never inferred
        self._run_log: list[str] = []
        self._run_result: dict | None = None
        # The live recorder for the run in progress. The dashboard polls it for
        # step progress, so the UI shows "step 4 of 9" instead of asking the
        # user to read a scrolling log.
        self._run_metrics: RunMetrics | None = None
        # One rendered PNG per screen. Kept OUT of the status payload and served
        # from its own endpoint: /api/status is polled every 1.5s, and shipping
        # five full-page screenshots on every poll is megabytes a minute.
        self._run_screens: list[dict] = []
        # Set when the user asks to stop. A threading.Event because it is set
        # from the HTTP thread and read from the run thread.
        self._stop = threading.Event()
        self._last_captured_generation = -1

    # -- connection watcher: builds the gallery over time, unattended -------

    def watch_connections(self) -> None:
        """Whenever a new plugin connection reports a file, snapshot it into
        the registry. Runs for the process's lifetime on its own thread.

        Nothing in here may be allowed to escape. A screenshot that fails --
        the plugin closed between connecting and being asked, Figma is in Dev
        Mode -- used to kill this thread outright, after which the gallery
        silently stopped updating for the rest of the process with no error
        anywhere the user could see.
        """
        while True:
            try:
                generation = self.bridge.connection_generation
                identity = self.bridge.current_file
                if generation != self._last_captured_generation and identity and identity.file_key:
                    self._capture(identity.file_key, identity.file_name)
                    # Only mark it done once it actually worked, so a transient
                    # failure is retried on the next tick rather than skipped.
                    self._last_captured_generation = generation
            except Exception as exc:
                logging.getLogger(__name__).info("Could not snapshot the connected file: %s", exc)
            time.sleep(CONNECT_POLL_INTERVAL)

    def _capture(self, file_key: str, file_name: str) -> None:
        """Store a screenshot of the connected canvas as `file_key`'s thumbnail.

        The screenshot always comes from whichever file the plugin is in RIGHT
        NOW, while `file_key` is the file we are about to file it under. Those
        are usually the same and occasionally not: the user switches file in
        Figma during a run, or the plugin reconnects between the caller reading
        the identity and this call. Filing it regardless is how one design's
        picture ended up on another design's card, so the identity is checked
        again AFTER the render and a mismatch is thrown away.
        """
        if not self._connected_to(file_key):
            return
        shot = get_screenshot(self.bridge)
        if not self._connected_to(file_key):
            return  # the plugin moved to another file while we were rendering
        thumbnail = shot["image_base64"] if shot["ok"] else None
        previous = self.registry.get(file_key)
        self.registry.upsert(
            FileEntry(
                file_key=file_key,
                file_name=file_name,
                # A failed screenshot is not evidence the design is gone. Keep
                # the last good picture rather than blanking the card -- upsert
                # replaces the whole entry, so writing None here erased it.
                thumbnail_base64=thumbnail or (previous.thumbnail_base64 if previous else None),
                last_seen=_now(),
            )
        )

    def _connected_to(self, file_key: str) -> bool:
        """Is the plugin, right now, in the file we think it is?"""
        identity = self.bridge.current_file
        return identity is not None and identity.file_key == file_key

    # -- run orchestration ---------------------------------------------------

    def stop_run(self) -> tuple[bool, str]:
        """Ask the run in progress to stop.

        Cooperative: a model call or a Figma round trip already in flight
        cannot be interrupted, so this promises that no NEW work starts. In
        practice the run ends within one model call, and whatever it built so
        far is kept -- a half-finished design is still the user's design.
        """
        with self._lock:
            # "stopping" counts as in progress: a second click on a button that
            # has not caught up yet should say so, not claim nothing is running.
            if self._run_status not in ("waiting_for_file", "running", "stopping"):
                return False, "There is no run in progress."
            if self._stop.is_set():
                return True, "Already stopping."
            self._stop.set()
            self._run_status = "stopping"
            self._run_log.append("Stopping -- waiting for the current step to finish...")
        return True, "Stopping after the current step."

    def start_run(
        self,
        file_key: str,
        instruction: str,
        mode: str = "create",
        attachments: list | None = None,
    ) -> tuple[bool, str]:
        """`mode` is "create" (build a new design) or "edit" (change this one).

        Declared by the user, never inferred from the wording. Guessing would
        mean an edit request misread as a build stamps a second screen beside
        the design it was supposed to change -- and undoing that is manual work
        in someone else's file.
        """
        if not self.settings_store.effective().is_model_configured:
            return False, "No model configured. Open Settings and add your API details first."
        with self._lock:
            if self._run_status in ("waiting_for_file", "running"):
                return False, "A run is already in progress."
            entry = self.registry.get(file_key)
            if entry is None:
                return False, f"Unknown file: {file_key}"
            self._run_status = "waiting_for_file"
            self._run_screens = []
            self._stop.clear()   # a new run is never born already stopping
            self._run_mode = "edit" if mode == "edit" else "create"
            self._run_log = [
                f"Waiting for '{entry.file_name}' to connect -- open it in Figma Desktop "
                "and run the plugin if it isn't already."
            ]
            if self._run_mode == "edit":
                self._run_log.append(
                    "Edit mode: select what you want changed in Figma, or leave nothing "
                    "selected and the agent will find it."
                )
            self._run_result = None
        threading.Thread(
            target=self._run_worker,
            args=(file_key, entry.file_name, instruction, self._run_mode, list(attachments or [])),
            daemon=True,
        ).start()
        return True, "started"

    def _run_worker(
        self,
        file_key: str,
        file_name: str,
        instruction: str,
        mode: str = "create",
        attachments: list | None = None,
    ) -> None:
        started_at = _now()
        run_metrics = RunMetrics()
        with self._lock:
            self._run_metrics = run_metrics

        # Time spent waiting for Figma is not the agent being slow, but it
        # dominates wall clock and gets blamed on the agent without this.
        waiting_since = time.monotonic()
        connected = self._wait_for_file(file_key)
        if self._stop.is_set():
            with self._lock:
                self._run_status = "stopped"
                self._run_log.append("Stopped before the run started.")
            self._record_history(
                file_key, file_name, instruction, started_at, None,
                error="Stopped before the run started.",
            )
            return
        run_metrics.plugin_wait_seconds = time.monotonic() - waiting_since
        if not connected:
            timeout_message = f"Timed out waiting for '{file_name}' to connect."
            with self._lock:
                self._run_status = "error"
                self._run_log.append(timeout_message)
            self._record_history(
                file_key, file_name, instruction, started_at, None, error=timeout_message
            )
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
            # Read the attachments BEFORE anything is built. An image needs a
            # model that can see it, and finding that out halfway through a run
            # means the design was already built without the reference.
            references = ""
            if attachments:
                for note in reference.check_readable(attachments, settings.has_vision):
                    logger.info("%s", note)
                logger.info("Reading %d attachment(s)...", len(attachments))
                references, attach_warnings = reference.describe(
                    attachments, build_vision_client(settings)
                )
                for warning in attach_warnings:
                    logger.info("%s", warning)
            if mode == "edit":
                # No vision critic and no repair pass: those judge whether a
                # design is good, and an edit run is not being asked to have an
                # opinion about the user's design -- only to make the change.
                result = edit_loop.run(
                    instruction,
                    self.bridge,
                    llm,
                    int(prefs["max_retries"]),
                    run_metrics=run_metrics,
                    should_stop=self._stop.is_set,
                    references=references,
                    attachments=attachments,
                )
            else:
                result = loop.run(
                    instruction,
                    self.bridge,
                    llm,
                    int(prefs["max_retries"]),
                    int(prefs["max_steps"]),
                    visual_gate=bool(prefs["visual_gate"]),
                    critic_llm=build_critic_client(settings),
                    run_metrics=run_metrics,
                    final_repair=bool(prefs["final_repair"]),
                    should_stop=self._stop.is_set,
                    references=references,
                    # Read as words for the brief AND uploaded as real images
                    # the design can place (agent/assets.py).
                    attachments=attachments,
                    prototype=bool(prefs["prototype"]),
                )
            # Best-effort: the design is already built, so a failed thumbnail
            # refresh must not report the whole run as crashed and throw the
            # result away.
            try:
                self._capture(file_key, file_name)
            except Exception as exc:
                logger.info("Could not refresh the file thumbnail: %s", exc)
            with self._lock:
                self._run_status = "stopped" if getattr(result, "stopped", False) else "done"
                self._run_mode = mode
                self._run_screens = list(getattr(result, "screen_shots", []) or [])
                self._run_result = _result_payload(result)
            self._record_history(file_key, file_name, instruction, started_at, result)
        except reference.ReferenceError as exc:
            # Nothing was built, and the reason is entirely actionable: say it
            # plainly rather than as a crash the user has to interpret.
            with self._lock:
                self._run_status = "error"
                self._run_log.append(str(exc))
            self._record_history(file_key, file_name, instruction, started_at, None, error=str(exc))
        except Exception as exc:  # the UI must hear about this, not spin forever
            with self._lock:
                self._run_status = "error"
                self._run_log.append(f"Run crashed: {exc}")
            self._record_history(
                file_key, file_name, instruction, started_at, None,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)

    def _record_history(
        self, file_key, file_name, instruction, started_at, result, error: str = ""
    ) -> None:
        """A run that failed is still worth remembering -- that's the point of a log.

        Records what the run DID, not just that it happened: how long it took,
        how much of the instruction it satisfied, what was left wrong. A row
        reading "Success - 0 nodes" is a log entry nobody can act on.
        """
        try:
            metrics = getattr(result, "metrics", None) or {} if result is not None else {}
            requirements_met = len(getattr(result, "requirements_met", []) or []) if result else 0
            requirements_missing = (
                len(getattr(result, "requirements_missing", []) or []) if result else 0
            )
            self.history.add(
                HistoryEntry(
                    id=uuid.uuid4().hex[:12],
                    instruction=instruction,
                    file_key=file_key,
                    file_name=file_name,
                    status=_history_status(result, error),
                    success=bool(result.success) if result is not None else False,
                    created_node_count=len(result.created_node_ids) if result is not None else 0,
                    failed_step_count=len(result.failed_steps) if result is not None else 0,
                    started_at=started_at,
                    finished_at=_now(),
                    thumbnail_base64=(result.final_screenshot_base64 if result is not None else None),
                    duration_seconds=float(metrics.get("elapsed_seconds") or 0.0),
                    section_count=int(metrics.get("steps_completed") or 0),
                    requirements_met=requirements_met,
                    requirements_total=requirements_met + requirements_missing,
                    layout_defect_count=len(getattr(result, "layout_defects", []) or []) if result else 0,
                    error=error,
                )
            )
        except Exception as exc:  # history must never break a run
            logging.getLogger(__name__).info("Could not record history: %s", exc)

    def _wait_for_file(self, file_key: str) -> bool:
        deadline = time.monotonic() + MAX_WAIT_FOR_FILE_SECONDS
        while time.monotonic() < deadline:
            if self._stop.is_set():
                return False   # waiting is the easiest thing of all to abandon
            identity = self.bridge.current_file
            if identity is not None and identity.file_key == file_key:
                return True
            time.sleep(CONNECT_POLL_INTERVAL)
        return False

    # -- removing a file from the gallery -------------------------------------

    def forget_file(self, file_key: str, clear_canvas: bool = False) -> tuple[bool, str]:
        """Drop a file from the gallery, optionally emptying its canvas first.

        There is deliberately no "delete the Figma file" here, because no such
        operation exists: a plugin runs INSIDE a file and the Plugin API has no
        `deleteFile` (checked against the real typings, and `figma.fileKey` is
        read-only). Figma's REST API has no delete-file endpoint either. The two
        things that ARE possible are offered instead, and named honestly.
        """
        with self._lock:
            if self._run_status in ("waiting_for_file", "running"):
                return False, "A run is in progress. Wait for it to finish first."

        entry = self.registry.get(file_key)
        if entry is None:
            return False, "That file is not in the gallery."

        message = f"Removed '{entry.file_name}' from the gallery."
        if clear_canvas:
            ok, detail = self.clear_canvas(file_key)
            if not ok:
                return False, detail
            message = f"{detail} {message}"

        self.registry.remove(file_key)
        return True, message

    def clear_canvas(self, file_key: str) -> tuple[bool, str]:
        """Empty the current page of the connected file. Destructive.

        Guarded by the same identity check the thumbnail capture uses: the
        script always runs in whichever file the plugin is in RIGHT NOW, so
        without re-checking, clicking delete on one card could wipe a different
        design entirely.
        """
        if not self._connected_to(file_key):
            return False, (
                "That file is not open with the plugin running, so its canvas cannot "
                "be touched. Open it in Figma Desktop and try again."
            )
        result = execute_figma_js(self.bridge, scaffold.build_clear_page_script())
        if not result["ok"]:
            return False, f"Could not clear the canvas: {result['error']}"
        payload = result.get("result") or {}
        removed = int(payload.get("removed") or 0)
        return True, f"Deleted {removed} top-level layer(s) from the canvas."

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
                "mode": self._run_mode,
                "log": list(self._run_log),
                "result": self._run_result,
                "connected_file": (
                    {"file_key": identity.file_key, "file_name": identity.file_name} if identity else None
                ),
                "plugin_connected": self.bridge.is_connected,
                "model_configured": self.settings_store.effective().is_model_configured,
                "metrics": self._live_metrics(),
            }

    def screen_image(self, index: int) -> bytes | None:
        """The PNG for one screen, or None if there is no such screen."""
        with self._lock:
            shots = list(self._run_screens)
        if not 0 <= index < len(shots):
            return None
        try:
            return base64.b64decode(shots[index].get("image_base64") or "")
        except (ValueError, TypeError):
            return None

    def _live_metrics(self) -> dict | None:
        """A snapshot of the run in progress, for the progress display.

        Read from the HTTP thread while the run thread is writing, so a torn
        read is possible in principle -- and a status endpoint that 500s
        because a counter moved would be a far worse bug than a progress bar
        that skips a frame.
        """
        if self._run_metrics is None:
            return None
        try:
            return self._run_metrics.snapshot()
        except Exception:
            return None

    # -- settings + setup ------------------------------------------------------

    def settings_snapshot(self) -> dict:
        view = self.settings_store.view()
        s = view.settings
        return {
            "model_base_url": s.model_base_url,
            "model_name": s.model_name,
            "vision_model_name": s.vision_model_name or s.critic_model_name,
            "has_vision": s.has_vision,
            "model_api_key_masked": mask(s.model_api_key),  # never send the real key
            "has_api_key": bool(s.model_api_key),
            "sources": view.sources,
            "configured": s.is_model_configured,
            "bridge_url": f"ws://{s.bridge_host}:{s.bridge_port}",
            "prefs": self.settings_store.prefs(),
            # The UI renders its inputs from this rather than hardcoding bounds
            # in HTML, so the limits it enforces are the ones the store applies.
            "prefs_schema": self.settings_store.schema(),
        }

    def history_snapshot(self) -> list[dict]:
        rows = [e.summary() for e in self.history.list_entries()]
        return self._fill_missing_thumbnails(rows)

    # Only the newest few runs keep their own screenshot (web/history.py caps it,
    # because a full-page PNG dwarfs the record it belongs to). Borrowing the
    # file's gallery thumbnail for the rest is bounded by the same reasoning:
    # enough rows to look like a log of real designs, not so many that opening
    # the tab ships megabytes of duplicated images.
    THUMBNAIL_FALLBACK_LIMIT = 6

    def _fill_missing_thumbnails(self, rows: list[dict]) -> list[dict]:
        """Show the file a run built in, when the run kept no screenshot itself.

        Flagged with `thumbnail_is_file` so the UI can be honest about it: this
        is the file as it looks NOW, not a picture of that particular run.
        """
        borrowed = 0
        by_key: dict[str, str | None] = {}
        for row in rows:
            if row.get("thumbnail_base64") or borrowed >= self.THUMBNAIL_FALLBACK_LIMIT:
                continue
            key = row.get("file_key")
            if key not in by_key:
                entry = self.registry.get(key) if key else None
                by_key[key] = entry.thumbnail_base64 if entry else None
            if by_key[key]:
                row["thumbnail_base64"] = by_key[key]
                row["thumbnail_is_file"] = True
                borrowed += 1
        return rows

    def update_prefs(self, values: dict) -> dict:
        """Apply preferences and report anything we would not accept.

        The errors ride back on the normal snapshot so the panel can show them
        next to the field, instead of silently reverting the input."""
        update = self.settings_store.update_prefs(values)
        snapshot = self.settings_snapshot()
        snapshot["pref_errors"] = update.errors
        return snapshot

    def update_settings(self, values: dict) -> dict:
        # An ABSENT field means "leave it alone"; an explicit empty string
        # means "clear it". Without that distinction a partial update (the UI
        # omits the API key when the user leaves it blank) would silently wipe
        # the fields it didn't mention.
        #
        # The editable set is imported, not restated: two copies of this tuple
        # meant adding a field to the store left the API quietly refusing it.
        self.settings_store.update({k: values[k] for k in EDITABLE if k in values})
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


def _history_status(result, error: str) -> str:
    """`stopped` is neither a success nor a crash, and reads as neither."""
    if result is None:
        return "stopped" if "Stopped" in (error or "") else "error"
    return "stopped" if getattr(result, "stopped", False) else "done"


def _result_payload(result) -> dict:
    """The finished run, as the dashboard's JSON.

    The reporting fields are read defensively. They are all real attributes of
    RunResult, but this dict is built AFTER the design exists on the canvas, and
    it sits inside the worker's `except Exception` -- so one missing attribute
    here used to turn a completed design into "Run crashed" and throw the whole
    result away. Losing real work to a missing statistic is not a trade worth
    making.
    """
    optional = (
        "layout_defects", "design_notes", "requirements_met", "requirements_missing",
        "screens", "interactions",
    )
    shots = getattr(result, "screen_shots", []) or []
    payload = {
        "success": result.success,
        "stopped": bool(getattr(result, "stopped", False)),
        # The plugin went away mid-run. Reported separately from a failure,
        # because "Figma disconnected" and "the design is wrong" are different
        # things to be told -- and the nodes it did build are really there.
        "ended_early": bool(getattr(result, "ended_early", False)),
        "created_node_count": len(result.created_node_ids),
        "failed_steps": result.failed_steps,
        "warnings": result.warnings,
        "metrics": getattr(result, "metrics", {}) or {},
        # NAMES only. The images are fetched one at a time from /api/screens, so
        # a finished run does not re-send every screenshot on every 1.5s poll --
        # five full-page PNGs in the status payload is megabytes a minute.
        "screen_names": [str(shot.get("name", "Screen")) for shot in shots],
    }
    payload.update({name: list(getattr(result, name, []) or []) for name in optional})
    return payload


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
            parsed = urlparse(self.path)
            if parsed.path == "/api/screens":
                self._send_screen(parsed.query)
                return
            if self.path in ("/", "/index.html"):
                self._send_static("index.html", "text/html")
            elif self.path in routes:
                self._send_json(200, routes[self.path]())
            else:
                self._send_json(404, {"error": "not found"})

        def _send_screen(self, query: str) -> None:
            """One screen's PNG, by index. Cacheable, so paging back and forth
            through a finished design does not re-fetch anything."""
            try:
                index = int((parse_qs(query).get("i") or ["0"])[0])
            except ValueError:
                index = 0
            image = dashboard.screen_image(index)
            if image is None:
                self._send_json(404, {"error": "no such screen"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(image)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(image)

        # An attached screenshot arrives base64-encoded inside the JSON body, so
        # the limit has to allow for one -- but it still has to BE a limit: this
        # server reads the whole body into memory before parsing it.
        MAX_BODY_BYTES = reference.MAX_TOTAL_BYTES * 2

        def _read_json(self) -> dict | None:
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                self._send_json(400, {"error": "bad Content-Length"})
                return None
            if length > self.MAX_BODY_BYTES:
                self._send_json(413, {"ok": False, "message": "That request is too large."})
                return None
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
            elif self.path == "/api/run/stop":
                ok, message = dashboard.stop_run()
                self._send_json(200 if ok else 409, {"ok": ok, "message": message})
            elif self.path == "/api/files/delete":
                self._handle_delete_file()
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
            mode = "edit" if (payload.get("mode") or "create") == "edit" else "create"
            # Attachments are decoded and size-checked HERE, so an unusable one
            # is a 400 the user can read rather than a run that starts, spends
            # model calls and then reports it could not open the file.
            try:
                attachments = reference.from_payload(payload.get("attachments") or [])
            except reference.ReferenceError as exc:
                self._send_json(400, {"ok": False, "message": str(exc)})
                return
            ok, message = dashboard.start_run(file_key, instruction, mode, attachments)
            self._send_json(202 if ok else 409, {"ok": ok, "message": message})

        def _handle_delete_file(self) -> None:
            payload = self._read_json()
            if payload is None:
                return
            file_key = payload.get("file_key")
            if not file_key:
                self._send_json(400, {"error": "file_key is required"})
                return
            ok, message = dashboard.forget_file(file_key, bool(payload.get("clear_canvas")))
            self._send_json(200 if ok else 409, {"ok": ok, "message": message})

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

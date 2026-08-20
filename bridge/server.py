"""Asyncio WebSocket server, id-based request/response matching.

The bridge is dumb: it moves messages between the agent and the Figma plugin
and matches responses to requests by id. No agent logic lives here.

The Figma plugin's `ui.html` connects to this server as a client. The rest of
the (synchronous) agent code calls `Bridge.send()` like an ordinary blocking
function; internally it hands the request to a background asyncio loop and
waits for the matching response.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import threading
import uuid
from dataclasses import dataclass

import websockets

from bridge.protocol import Request, Response

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0

# Biggest single WebSocket message, in either direction. Screenshots of a tall
# page and attached images are both megabytes; the library's 1MB default closes
# the socket on one.
MAX_MESSAGE_BYTES = 64 * 1024 * 1024


class BridgeError(RuntimeError):
    """Raised when a request can't be delivered or answered (no plugin, timeout, disconnect)."""


def new_request_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class FileIdentity:
    """Which Figma file the currently-connected plugin instance is running in."""

    file_key: str | None  # None if the running plugin predates the hello handshake
    file_name: str


class Bridge:
    """WebSocket server the Figma plugin connects to as a client."""

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server: websockets.WebSocketServer | None = None
        self._connection: websockets.WebSocketServerProtocol | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._connected_event = threading.Event()
        self._current_file: FileIdentity | None = None
        self._connection_generation = 0

    # -- lifecycle ------------------------------------------------------

    def start(self) -> None:
        """Start the server on a background thread; block until it's listening."""
        ready = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, args=(ready,), daemon=True)
        self._thread.start()
        ready.wait()

    def _run_loop(self, ready: threading.Event) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve(ready))
        self._loop.run_forever()

    async def _serve(self, ready: threading.Event) -> None:
        # Keepalive pings are disabled deliberately. Figma throttles a plugin's
        # UI iframe when its window isn't focused, so the plugin often can't
        # answer a ping inside the default 20s timeout -- the server then kills
        # a perfectly healthy connection ("no close frame received or sent"),
        # ui.html reconnects, and the cycle repeats. This link is loopback-only
        # and every request already has its own timeout, so we don't need pings
        # to detect a dead peer.
        # The 1MB default is far too small for what actually crosses this
        # link: a full-page PNG comes back base64-encoded, and an attached
        # photo goes out the same way inside the script that uploads it. Over
        # the limit the library closes the connection rather than erroring, so
        # it surfaces as "Figma disconnected" mid-run.
        self._server = await websockets.serve(
            self._handler, self._host, self._port, ping_interval=None,
            max_size=MAX_MESSAGE_BYTES,
        )
        ready.set()

    def stop(self) -> None:
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop).result(timeout=5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)

    async def _shutdown(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    def wait_for_plugin(self, timeout: float = 120.0) -> bool:
        """Block until the Figma plugin connects (or the timeout elapses)."""
        return self._connected_event.wait(timeout)

    @property
    def is_connected(self) -> bool:
        return self._connection is not None

    @property
    def current_file(self) -> FileIdentity | None:
        """Identity of the file the currently-connected plugin instance reported, if any."""
        return self._current_file

    @property
    def connection_generation(self) -> int:
        """Increments every time a new plugin connection is established.

        Lets a poller (see web/app.py) detect "a new connection just happened"
        without racing on file_key alone (e.g. reconnecting to the same file).
        """
        return self._connection_generation

    @property
    def port(self) -> int:
        """The actual bound port (useful when constructed with port=0 in tests)."""
        assert self._server is not None, "Bridge is not started"
        return self._server.sockets[0].getsockname()[1]

    # -- message handling -------------------------------------------------

    async def _handler(self, websocket) -> None:
        self._connection = websocket
        self._connection_generation += 1
        self._connected_event.set()
        await self._handshake(websocket)
        try:
            async for raw in websocket:
                self._on_message(raw)
        except websockets.exceptions.ConnectionClosed:
            # Normal: the plugin was closed, re-run, or the file switched.
            # ui.html reconnects on its own, so this is not an error -- letting
            # it propagate makes the websockets library dump a scary traceback
            # for a routine disconnect.
            logger.info("Figma plugin disconnected; waiting for it to reconnect.")
        finally:
            if self._connection is websocket:
                self._connection = None
                self._current_file = None
                self._connected_event.clear()
            self._fail_all_pending("Figma plugin disconnected")

    async def _handshake(self, websocket) -> None:
        """Ask the just-connected plugin which file it's running in.

        Talks to the socket directly (send + recv), not via `_send`/`_pending`:
        the normal `async for raw in websocket` reader hasn't started yet at
        this point in `_handler`, so routing through the pending-futures path
        here would deadlock waiting for a reader that hasn't started.

        Best-effort: any failure (old plugin build, bad reply, timeout) just
        leaves current_file as None instead of breaking the connection.
        """
        request = Request(id=new_request_id(), type="hello")
        try:
            await websocket.send(json.dumps(request.to_json()))
            raw = await asyncio.wait_for(websocket.recv(), timeout=5.0)
        except Exception:
            return
        try:
            response = Response.from_json(json.loads(raw))
        except (json.JSONDecodeError, KeyError):
            return
        if response.id == request.id and response.ok and isinstance(response.result, dict):
            self._current_file = FileIdentity(
                file_key=response.result.get("fileKey"),
                file_name=response.result.get("fileName") or "Untitled",
            )

    def _on_message(self, raw: str) -> None:
        response = Response.from_json(json.loads(raw))
        future = self._pending.pop(response.id, None)
        if future is not None and not future.done():
            future.set_result(response)

    def _fail_all_pending(self, message: str) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(BridgeError(message))
        self._pending.clear()

    # -- public API ---------------------------------------------------------

    def send(self, request: Request, timeout: float = DEFAULT_TIMEOUT) -> Response:
        """Send a request and block for the matching response. Thread-safe."""
        if self._loop is None:
            raise BridgeError("Bridge is not started")
        future = asyncio.run_coroutine_threadsafe(self._send(request), self._loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            self._loop.call_soon_threadsafe(self._cancel_pending, request.id)
            raise BridgeError(f"Timed out waiting for plugin response to {request.id}") from exc

    def _cancel_pending(self, request_id: str) -> None:
        # Wake up the still-suspended _send() coroutine so its task doesn't
        # linger past this call and get destroyed while pending at shutdown.
        future = self._pending.pop(request_id, None)
        if future is not None and not future.done():
            future.set_exception(BridgeError("Timed out waiting for plugin response"))

    async def _send(self, request: Request) -> Response:
        if self._connection is None:
            raise BridgeError("No Figma plugin connected. Open the plugin in Figma Desktop.")
        pending: asyncio.Future = self._loop.create_future()
        self._pending[request.id] = pending
        await self._connection.send(json.dumps(request.to_json()))
        return await pending

"""Protocol round-trips. No Figma required -- a background asyncio client
stands in for the plugin over a real loopback WebSocket.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import asdict

import pytest
import websockets

from bridge.protocol import Request, Response
from bridge.server import Bridge, BridgeError


def test_request_response_json_round_trip():
    """Pure (de)serialization -- no sockets at all."""
    request = Request(id="abc123", type="exec", code="return {createdNodeIds: []}")
    assert request.to_json() == {
        "id": "abc123",
        "type": "exec",
        "code": "return {createdNodeIds: []}",
        "node_id": None,
    }

    response = Response.from_json(
        {"id": "abc123", "ok": True, "result": {"createdNodeIds": ["1:2"]}}
    )
    assert response == Response(id="abc123", ok=True, result={"createdNodeIds": ["1:2"]})


class FakePlugin:
    """A minimal WebSocket client standing in for the Figma plugin's ui.html.

    Answers the bridge's initial "hello" handshake automatically (like the
    real plugin does), then replies to exactly one subsequent request with
    `reply`.
    """

    def __init__(
        self,
        port: int,
        reply: Response,
        file_key: str = "test-file-key",
        file_name: str = "Test File",
    ):
        self._port = port
        self._reply = reply
        self._file_key = file_key
        self._file_name = file_name
        self.received: dict | None = None
        self.ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        async with websockets.connect(f"ws://127.0.0.1:{self._port}") as ws:
            self.ready.set()

            hello = json.loads(await ws.recv())
            hello_reply = Response(
                id=hello["id"], ok=True, result={"fileKey": self._file_key, "fileName": self._file_name}
            )
            await ws.send(json.dumps(asdict(hello_reply)))

            raw = await ws.recv()
            self.received = json.loads(raw)
            reply = Response(id=self.received["id"], ok=self._reply.ok, result=self._reply.result)
            await ws.send(json.dumps(asdict(reply)))
            await asyncio.sleep(2)  # keep the socket open while the test asserts


@pytest.fixture
def bridge():
    b = Bridge("127.0.0.1", 0)  # port 0 -> OS assigns a free port
    b.start()
    yield b
    b.stop()


def test_send_matches_response_by_id(bridge):
    assert bridge.wait_for_plugin(timeout=0.2) is False  # nothing connected yet

    plugin = FakePlugin(bridge.port, reply=Response(id="", ok=True, result={"createdNodeIds": ["1:2"]}))
    plugin.start()
    assert plugin.ready.wait(timeout=5)
    assert bridge.wait_for_plugin(timeout=5) is True
    assert bridge.is_connected is True

    request = Request(id="req-1", type="exec", code="return {createdNodeIds: ['1:2']}")
    response = bridge.send(request, timeout=5)

    assert response.ok is True
    assert response.result == {"createdNodeIds": ["1:2"]}
    assert plugin.received["type"] == "exec"
    assert plugin.received["id"] == "req-1"

    # The handshake ran automatically before this request was ever sent.
    assert bridge.current_file is not None
    assert bridge.current_file.file_key == "test-file-key"
    assert bridge.current_file.file_name == "Test File"


def test_current_file_clears_on_disconnect(bridge):
    plugin = FakePlugin(bridge.port, reply=Response(id="", ok=True, result={"createdNodeIds": []}))
    plugin.start()
    assert plugin.ready.wait(timeout=5)
    assert bridge.wait_for_plugin(timeout=5) is True
    bridge.send(Request(id="req-x", type="ping"), timeout=5)
    assert bridge.current_file is not None

    # FakePlugin's _serve() coroutine closes its socket ~2s after replying,
    # which should propagate to the bridge clearing current_file.
    for _ in range(50):
        if bridge.current_file is None:
            break
        time.sleep(0.1)
    assert bridge.current_file is None


def test_send_without_a_connected_plugin_raises(bridge):
    with pytest.raises(BridgeError):
        bridge.send(Request(id="req-2", type="ping"), timeout=0.5)


def test_send_times_out_if_plugin_never_replies(bridge):
    async def silent_client(port: int, ready: threading.Event):
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            ready.set()
            await ws.recv()  # receive the handshake's "hello" but never reply to anything
            await asyncio.sleep(2)

    ready = threading.Event()
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=lambda: loop.run_until_complete(silent_client(bridge.port, ready)), daemon=True)
    thread.start()
    assert ready.wait(timeout=5)
    assert bridge.wait_for_plugin(timeout=5) is True

    with pytest.raises(BridgeError):
        bridge.send(Request(id="req-3", type="ping"), timeout=0.5)

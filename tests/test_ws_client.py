"""Tests for ws_client.py — VoiceLiveClient protocol helpers."""

import json
from unittest.mock import MagicMock, patch

import pytest

from ws_client import VoiceLiveClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(**kwargs) -> VoiceLiveClient:
    return VoiceLiveClient(
        endpoint="wss://example.services.ai.azure.com/voice-live/realtime?api-version=2025-10-01",
        **kwargs,
    )


class FakeWebSocket:
    """Minimal stand-in for websocket.WebSocket."""

    def __init__(self, recv_messages: list[str] | None = None):
        self._recv_messages = list(recv_messages or [])
        self._recv_idx = 0
        self.sent: list[str] = []
        self._timeout = None
        self.closed = False

    def send(self, data: str):
        self.sent.append(data)

    def recv(self) -> str:
        if self._recv_idx >= len(self._recv_messages):
            raise TimeoutError("No more messages")
        msg = self._recv_messages[self._recv_idx]
        self._recv_idx += 1
        return msg

    def gettimeout(self):
        return self._timeout

    def settimeout(self, t):
        self._timeout = t

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConnect:
    @patch("ws_client.websocket.create_connection")
    def test_connect_with_bearer(self, mock_cc):
        fake_ws = FakeWebSocket()
        mock_cc.return_value = fake_ws

        client = _make_client(bearer_token="tok123")
        elapsed = client.connect()

        assert elapsed >= 0
        assert client.ws is fake_ws
        call_kwargs = mock_cc.call_args
        headers = call_kwargs.kwargs.get("header", call_kwargs[1].get("header", []))
        assert any("Bearer tok123" in h for h in headers)

    @patch("ws_client.websocket.create_connection")
    def test_connect_with_api_key(self, mock_cc):
        fake_ws = FakeWebSocket()
        mock_cc.return_value = fake_ws

        client = _make_client(api_key="key456")
        client.connect()

        url_arg = mock_cc.call_args[0][0]
        assert "api-key=key456" in url_arg


class TestSendEvent:
    def test_send_event_serializes_json(self):
        client = _make_client()
        client.ws = FakeWebSocket()

        client.send_event({"type": "test", "value": 42})
        sent = json.loads(client.ws.sent[0])
        assert sent == {"type": "test", "value": 42}

    def test_send_event_not_connected_raises(self):
        client = _make_client()
        with pytest.raises(RuntimeError, match="not connected"):
            client.send_event({"type": "test"})


class TestRecvEvent:
    def test_recv_parses_json(self):
        client = _make_client()
        msg = json.dumps({"type": "session.created"})
        client.ws = FakeWebSocket(recv_messages=[msg])

        event = client.recv_event()
        assert event["type"] == "session.created"

    def test_recv_not_connected_raises(self):
        client = _make_client()
        with pytest.raises(RuntimeError, match="not connected"):
            client.recv_event()


class TestSessionHelpers:
    def test_wait_for_session_created_returns_event(self):
        client = _make_client()
        msgs = [
            json.dumps({"type": "other"}),
            json.dumps({"type": "session.created", "session": {}}),
        ]
        client.ws = FakeWebSocket(recv_messages=msgs)

        event = client.wait_for_session_created(timeout=5)
        assert event["type"] == "session.created"

    def test_send_session_update_contents(self):
        client = _make_client()
        client.ws = FakeWebSocket()

        client.send_session_update(
            modalities=["text", "audio"],
            sample_rate=24000,
            voice="alloy",
        )

        sent = json.loads(client.ws.sent[0])
        assert sent["type"] == "session.update"
        assert sent["session"]["modalities"] == ["text", "audio"]
        assert sent["session"]["input_audio_sampling_rate"] == 24000
        assert sent["session"]["voice"] == "alloy"


class TestAudioHelpers:
    def test_send_audio_chunk(self):
        client = _make_client()
        client.ws = FakeWebSocket()

        client.send_audio_chunk("AQID")
        sent = json.loads(client.ws.sent[0])
        assert sent["type"] == "input_audio_buffer.append"
        assert sent["audio"] == "AQID"

    def test_send_commit(self):
        client = _make_client()
        client.ws = FakeWebSocket()

        client.send_commit()
        sent = json.loads(client.ws.sent[0])
        assert sent["type"] == "input_audio_buffer.commit"

    def test_send_clear(self):
        client = _make_client()
        client.ws = FakeWebSocket()

        client.send_clear()
        sent = json.loads(client.ws.sent[0])
        assert sent["type"] == "input_audio_buffer.clear"

    def test_send_response_create(self):
        client = _make_client()
        client.ws = FakeWebSocket()

        client.send_response_create(modalities=["text"])
        sent = json.loads(client.ws.sent[0])
        assert sent["type"] == "response.create"
        assert sent["response"]["modalities"] == ["text"]

    def test_send_response_create_no_modalities(self):
        client = _make_client()
        client.ws = FakeWebSocket()

        client.send_response_create()
        sent = json.loads(client.ws.sent[0])
        assert sent["type"] == "response.create"
        assert "response" not in sent


class TestClose:
    def test_close_sets_ws_none(self):
        client = _make_client()
        fake_ws = FakeWebSocket()
        client.ws = fake_ws

        client.close()
        assert client.ws is None
        assert fake_ws.closed

    def test_close_when_already_none(self):
        client = _make_client()
        client.close()  # should not raise

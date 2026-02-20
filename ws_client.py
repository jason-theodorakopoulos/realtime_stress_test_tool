"""WebSocket client wrapper for the Voice Live API."""

import json
import logging
import time
from typing import Any

import websocket  # websocket-client library (gevent-compatible)

logger = logging.getLogger(__name__)


class VoiceLiveClient:
    """Thin wrapper around a WebSocket connection to the Voice Live API."""

    def __init__(
        self,
        endpoint: str,
        bearer_token: str | None = None,
        api_key: str | None = None,
        connect_timeout: float = 10.0,
    ):
        self.endpoint = endpoint
        self.bearer_token = bearer_token
        self.api_key = api_key
        self.connect_timeout = connect_timeout
        self.ws: websocket.WebSocket | None = None

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def connect(self) -> float:
        """Open the WebSocket.  Returns elapsed connect time in seconds."""
        headers: list[str] = []
        url = self.endpoint

        if self.bearer_token:
            headers.append(f"Authorization: Bearer {self.bearer_token}")
        elif self.api_key:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}api-key={self.api_key}"

        t0 = time.perf_counter()
        self.ws = websocket.create_connection(
            url,
            header=headers,
            timeout=self.connect_timeout,
        )
        elapsed = time.perf_counter() - t0
        logger.debug("WS connected in %.3fs", elapsed)
        return elapsed

    def close(self) -> None:
        """Gracefully close the WebSocket."""
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                logger.debug("Error closing WS", exc_info=True)
            finally:
                self.ws = None

    # ------------------------------------------------------------------
    # Send / receive
    # ------------------------------------------------------------------

    def send_event(self, event: dict[str, Any]) -> None:
        """Serialize *event* to JSON and send it over the WebSocket."""
        if self.ws is None:
            raise RuntimeError("WebSocket is not connected")
        payload = json.dumps(event, separators=(",", ":"))
        self.ws.send(payload)
        event_type = event.get("type", "?")
        if event_type != "input_audio_buffer.append":
            logger.debug("-> %s", event_type)

    def recv_event(self, timeout: float | None = None) -> dict[str, Any]:
        """Block until a JSON message arrives.  Returns parsed dict.

        Raises ``TimeoutError`` if *timeout* seconds elapse with no message.
        """
        if self.ws is None:
            raise RuntimeError("WebSocket is not connected")
        old_timeout = self.ws.gettimeout()
        try:
            if timeout is not None:
                self.ws.settimeout(timeout)
            raw = self.ws.recv()
        except websocket.WebSocketTimeoutException as exc:
            raise TimeoutError("No message within timeout") from exc
        finally:
            self.ws.settimeout(old_timeout)

        data: dict[str, Any] = json.loads(raw)
        event_type = data.get("type", "?")
        if event_type not in ("response.audio.delta", "response.text.delta"):
            logger.debug("<- %s", event_type)
        return data

    # ------------------------------------------------------------------
    # High-level protocol helpers
    # ------------------------------------------------------------------

    def wait_for_session_created(self, timeout: float = 10.0) -> dict[str, Any]:
        """Read events until ``session.created`` arrives."""
        deadline = time.perf_counter() + timeout
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for session.created")
            event = self.recv_event(timeout=remaining)
            if event.get("type") == "session.created":
                return event

    def send_session_update(
        self,
        modalities: list[str],
        input_audio_format: str = "pcm16",
        output_audio_format: str = "pcm16",
        sample_rate: int = 24000,
        voice: str | None = None,
        turn_detection_type: str | None = None,
        create_response: bool = False,
    ) -> None:
        """Send ``session.update`` with given parameters."""
        session: dict[str, Any] = {
            "modalities": modalities,
            "input_audio_format": input_audio_format,
            "output_audio_format": output_audio_format,
            "input_audio_sampling_rate": sample_rate,
            "turn_detection": {
                "create_response": create_response,
            },
        }
        if turn_detection_type:
            session["turn_detection"]["type"] = turn_detection_type
        if voice:
            session["voice"] = voice

        self.send_event({"type": "session.update", "session": session})

    def wait_for_session_updated(self, timeout: float = 10.0) -> dict[str, Any]:
        """Read events until ``session.updated`` arrives."""
        deadline = time.perf_counter() + timeout
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for session.updated")
            event = self.recv_event(timeout=remaining)
            if event.get("type") == "session.updated":
                return event

    def send_audio_chunk(self, b64_audio: str) -> None:
        """Send an ``input_audio_buffer.append`` event."""
        self.send_event({"type": "input_audio_buffer.append", "audio": b64_audio})

    def send_commit(self) -> None:
        """Send ``input_audio_buffer.commit``."""
        self.send_event({"type": "input_audio_buffer.commit"})

    def send_clear(self) -> None:
        """Send ``input_audio_buffer.clear``."""
        self.send_event({"type": "input_audio_buffer.clear"})

    def send_response_create(self, modalities: list[str] | None = None) -> None:
        """Send ``response.create``."""
        event: dict[str, Any] = {"type": "response.create"}
        if modalities:
            event["response"] = {"modalities": modalities}
        self.send_event(event)

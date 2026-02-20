"""Locust stress-test for Azure AI Speech Voice Live API (WebSocket).

Each Locust user opens one WebSocket, configures a session, then streams
a sequence of prerecorded PCM audio files as separate "user turns",
measuring connect latency, per-turn TTFB, and per-turn end-to-end time.
"""

import logging
import os
import time

import gevent
import locust
from locust import User, between, events, task

from audio import chunk_audio, encode_chunk, load_pcm_files
from ws_client import VoiceLiveClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

VOICELIVE_ENDPOINT = os.environ.get("VOICELIVE_ENDPOINT", "")
VOICELIVE_BEARER_TOKEN = os.environ.get("VOICELIVE_BEARER_TOKEN", "")
VOICELIVE_API_KEY = os.environ.get("VOICELIVE_API_KEY", "")
AUDIO_DIR = os.environ.get("AUDIO_DIR", "./audio_pcm24k")
TURNS_PER_CONVO = int(os.environ.get("TURNS_PER_CONVO", "4"))
CHUNK_MS = int(os.environ.get("CHUNK_MS", "20"))
STREAM_SPEED = float(os.environ.get("STREAM_SPEED", "1.0"))
RESPONSE_MODALITIES = os.environ.get("RESPONSE_MODALITIES", "text,audio").split(",")
TURN_THINK_TIME_MS = int(os.environ.get("TURN_THINK_TIME_MS", "0"))
WS_CONNECT_TIMEOUT_S = float(os.environ.get("WS_CONNECT_TIMEOUT_S", "10"))
TURN_TIMEOUT_S = float(os.environ.get("TURN_TIMEOUT_S", "30"))
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "24000"))
VOICE = os.environ.get("VOICE", "")
TURN_DETECTION_TYPE = os.environ.get("TURN_DETECTION_TYPE", "")

# Preload audio files once (shared across greenlets — read-only).
_PCM_BUFFERS: list[bytes] = []


@events.init.add_listener
def _on_locust_init(environment: locust.env.Environment, **_kwargs):
    """Load PCM files at Locust startup so each user doesn't re-read disk."""
    global _PCM_BUFFERS  # noqa: PLW0603
    try:
        _PCM_BUFFERS = load_pcm_files(AUDIO_DIR)
        logger.info(
            "Loaded %d PCM files from %s (total %.1f KB)",
            len(_PCM_BUFFERS),
            AUDIO_DIR,
            sum(len(b) for b in _PCM_BUFFERS) / 1024,
        )
    except FileNotFoundError:
        logger.warning(
            "Audio directory %s not found or empty — "
            "users will fail on first conversation.",
            AUDIO_DIR,
        )


def _fire(
    environment,
    request_type: str,
    name: str,
    response_time_ms: float,
    response_length: int = 0,
    exception: BaseException | None = None,
):
    """Convenience wrapper around ``events.request.fire``."""
    environment.events.request.fire(
        request_type=request_type,
        name=name,
        response_time=response_time_ms,
        response_length=response_length,
        exception=exception,
    )


# ---------------------------------------------------------------------------
# Locust User
# ---------------------------------------------------------------------------


class VoiceLiveUser(User):
    """Simulates one voice conversation at a time against the Voice Live API."""

    wait_time = between(1, 3)

    @task
    def conversation(self):
        """Run a full multi-turn conversation."""
        if not VOICELIVE_ENDPOINT:
            _fire(self.environment, "WS", "connect", 0, exception=RuntimeError("VOICELIVE_ENDPOINT not set"))
            return

        if not _PCM_BUFFERS:
            _fire(self.environment, "WS", "connect", 0, exception=RuntimeError("No PCM audio loaded"))
            return

        client = VoiceLiveClient(
            endpoint=VOICELIVE_ENDPOINT,
            bearer_token=VOICELIVE_BEARER_TOKEN or None,
            api_key=VOICELIVE_API_KEY or None,
            connect_timeout=WS_CONNECT_TIMEOUT_S,
        )

        convo_t0 = time.perf_counter()

        # ---- Connect ----
        try:
            connect_s = client.connect()
            _fire(self.environment, "WS", "connect", connect_s * 1000)
        except Exception as exc:
            _fire(self.environment, "WS", "connect", 0, exception=exc)
            return

        try:
            # ---- Wait for session.created ----
            try:
                client.wait_for_session_created(timeout=WS_CONNECT_TIMEOUT_S)
            except Exception as exc:
                _fire(self.environment, "WS", "session_created", 0, exception=exc)
                return

            # ---- Send session.update ----
            client.send_session_update(
                modalities=RESPONSE_MODALITIES,
                sample_rate=SAMPLE_RATE,
                voice=VOICE or None,
                turn_detection_type=TURN_DETECTION_TYPE or None,
                create_response=False,
            )

            try:
                client.wait_for_session_updated(timeout=WS_CONNECT_TIMEOUT_S)
            except Exception as exc:
                _fire(self.environment, "WS", "session_updated", 0, exception=exc)
                return

            # ---- Stream turns ----
            num_turns = min(TURNS_PER_CONVO, len(_PCM_BUFFERS))
            for turn_idx in range(num_turns):
                self._run_turn(client, turn_idx)

                if TURN_THINK_TIME_MS > 0:
                    gevent.sleep(TURN_THINK_TIME_MS / 1000.0)

            # ---- Conversation total ----
            convo_elapsed = (time.perf_counter() - convo_t0) * 1000
            _fire(self.environment, "WS", "conversation_total", convo_elapsed)

        finally:
            client.close()

    # ------------------------------------------------------------------

    def _run_turn(self, client: VoiceLiveClient, turn_idx: int):
        """Stream one audio file, commit, request response, wait for done."""
        pcm_data = _PCM_BUFFERS[turn_idx % len(_PCM_BUFFERS)]
        chunks = chunk_audio(pcm_data, CHUNK_MS, sample_rate=SAMPLE_RATE)

        # A) clear buffer
        client.send_clear()

        # B) stream audio chunks
        sleep_s = 0.0
        if STREAM_SPEED > 0:
            sleep_s = (CHUNK_MS / 1000.0) / STREAM_SPEED

        for chunk in chunks:
            client.send_audio_chunk(encode_chunk(chunk))
            if sleep_s > 0:
                gevent.sleep(sleep_s)

        # C) commit
        client.send_commit()
        t_commit = time.perf_counter()

        # D) request response
        client.send_response_create(modalities=RESPONSE_MODALITIES)

        # E) read server events until response.done
        t_first_delta: float | None = None
        deadline = time.perf_counter() + TURN_TIMEOUT_S
        turn_name = f"turn_{turn_idx}"

        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                exc = TimeoutError(f"Turn {turn_idx} timed out")
                _fire(self.environment, "WS", f"{turn_name}_ttfb", 0, exception=exc)
                _fire(self.environment, "WS", f"{turn_name}_e2e", 0, exception=exc)
                return

            try:
                event = client.recv_event(timeout=remaining)
            except Exception as exc:
                _fire(self.environment, "WS", f"{turn_name}_ttfb", 0, exception=exc)
                _fire(self.environment, "WS", f"{turn_name}_e2e", 0, exception=exc)
                return

            etype = event.get("type", "")

            if etype == "error":
                exc = RuntimeError(f"Server error: {event}")
                _fire(self.environment, "WS", f"{turn_name}_ttfb", 0, exception=exc)
                _fire(self.environment, "WS", f"{turn_name}_e2e", 0, exception=exc)
                return

            if etype in ("response.text.delta", "response.audio.delta"):
                if t_first_delta is None:
                    t_first_delta = time.perf_counter()
                    ttfb_ms = (t_first_delta - t_commit) * 1000
                    _fire(self.environment, "WS", f"{turn_name}_ttfb", ttfb_ms)

            if etype == "response.done":
                e2e_ms = (time.perf_counter() - t_commit) * 1000
                _fire(self.environment, "WS", f"{turn_name}_e2e", e2e_ms)
                return

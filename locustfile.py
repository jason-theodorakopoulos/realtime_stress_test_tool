"""Locust stress-test for Azure AI Speech Voice Live API (WebSocket).

Each Locust user opens one WebSocket, configures a session, then streams
a sequence of prerecorded PCM audio files as separate "user turns",
measuring connect latency, per-turn TTFB, and per-turn end-to-end time.
"""

import datetime
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
# Transcript file logger
# ---------------------------------------------------------------------------

LOG_DIR = os.environ.get("TRANSCRIPT_LOG_DIR", "./logs")


def _get_transcript_logger() -> logging.Logger:
    """Return a dedicated logger that writes transcripts to a timestamped file."""
    tlog = logging.getLogger("transcript")
    if not tlog.handlers:
        os.makedirs(LOG_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(os.path.join(LOG_DIR, f"{ts}.log"), encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        tlog.addHandler(fh)
        tlog.setLevel(logging.INFO)
    return tlog


transcript_logger: logging.Logger | None = None

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
TURN_DETECTION_TYPE = os.environ.get("TURN_DETECTION_TYPE", "server_vad")
INPUT_LANGUAGE = os.environ.get("INPUT_LANGUAGE", "")
INSTRUCTIONS = os.environ.get("INSTRUCTIONS", "")

# Preload audio files once (shared across greenlets — read-only).
_PCM_BUFFERS: list[bytes] = []
_PCM_NAMES: list[str] = []


@events.init.add_listener
def _on_locust_init(environment: locust.env.Environment, **_kwargs):
    """Load PCM files at Locust startup so each user doesn't re-read disk."""
    global _PCM_BUFFERS, _PCM_NAMES, transcript_logger  # noqa: PLW0603
    transcript_logger = _get_transcript_logger()
    try:
        _PCM_BUFFERS = load_pcm_files(AUDIO_DIR)
        # Store filenames for transcript logging
        import glob as _glob
        _PCM_NAMES = [
            os.path.basename(p)
            for p in sorted(_glob.glob(os.path.join(AUDIO_DIR, "*.raw")))
        ]
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
        convo_id = datetime.datetime.now().strftime("%H%M%S%f")

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
                create_response=True,
                input_audio_language=INPUT_LANGUAGE or None,
                instructions=INSTRUCTIONS or None,
            )

            try:
                client.wait_for_session_updated(timeout=WS_CONNECT_TIMEOUT_S)
            except Exception as exc:
                _fire(self.environment, "WS", "session_updated", 0, exception=exc)
                return

            # ---- Stream turns ----
            num_turns = min(TURNS_PER_CONVO, len(_PCM_BUFFERS))
            if transcript_logger:
                transcript_logger.info("[convo=%s] START (%d turns)", convo_id, num_turns)
            for turn_idx in range(num_turns):
                self._run_turn(client, turn_idx, convo_id)

                if TURN_THINK_TIME_MS > 0:
                    gevent.sleep(TURN_THINK_TIME_MS / 1000.0)

            if transcript_logger:
                transcript_logger.info("[convo=%s] END", convo_id)

            # ---- Graceful session close ----
            try:
                close_s = client.close_session()
                _fire(self.environment, "WS", "session_close", close_s * 1000)
            except Exception as exc:
                _fire(self.environment, "WS", "session_close", 0, exception=exc)

            # ---- Conversation total ----
            convo_elapsed = (time.perf_counter() - convo_t0) * 1000
            _fire(self.environment, "WS", "conversation_total", convo_elapsed)

        finally:
            client.close()

    # ------------------------------------------------------------------

    def _run_turn(self, client: VoiceLiveClient, turn_idx: int, convo_id: str = ""):
        """Stream one audio file and wait for the model's response.

        With server VAD (default), the server automatically detects speech
        boundaries, commits the audio buffer, and creates a response.
        Without VAD (``TURN_DETECTION_TYPE=none``), we manually commit and
        request a response after streaming.
        """
        buf_idx = turn_idx % len(_PCM_BUFFERS)
        pcm_data = _PCM_BUFFERS[buf_idx]
        audio_name = _PCM_NAMES[buf_idx] if buf_idx < len(_PCM_NAMES) else f"audio_{buf_idx}"
        chunks = chunk_audio(pcm_data, CHUNK_MS, sample_rate=SAMPLE_RATE)
        vad_enabled = TURN_DETECTION_TYPE.lower() not in ("", "none")

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

        t_audio_done = time.perf_counter()

        # C) Without VAD: manually commit and request a response.
        #    With VAD: append silence so VAD detects end-of-speech,
        #    then the server auto-commits and auto-creates a response.
        if not vad_enabled:
            client.send_commit()
            client.send_response_create(modalities=RESPONSE_MODALITIES)
        else:
            # Append ~700 ms of silence to trigger VAD end-of-speech.
            silence_bytes = int(SAMPLE_RATE * 2 * 0.7)  # PCM16 mono
            silence_chunk = encode_chunk(b"\x00" * silence_bytes)
            client.send_audio_chunk(silence_chunk)

        # D) read server events until response.done (with completed status)
        t_first_delta: float | None = None
        deadline = time.perf_counter() + TURN_TIMEOUT_S
        turn_name = f"turn_{turn_idx}"
        assistant_text_parts: list[str] = []
        user_transcript_parts: list[str] = []

        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                exc = TimeoutError(f"Turn {turn_idx} timed out")
                if t_first_delta is None:
                    _fire(self.environment, "WS", f"{turn_name}_ttfb", 0, exception=exc)
                _fire(self.environment, "WS", f"{turn_name}_e2e", 0, exception=exc)
                return

            try:
                event = client.recv_event(timeout=remaining)
            except Exception as exc:
                if t_first_delta is None:
                    _fire(self.environment, "WS", f"{turn_name}_ttfb", 0, exception=exc)
                _fire(self.environment, "WS", f"{turn_name}_e2e", 0, exception=exc)
                return

            etype = event.get("type", "")

            if etype == "error":
                exc = RuntimeError(f"Server error: {event}")
                if t_first_delta is None:
                    _fire(self.environment, "WS", f"{turn_name}_ttfb", 0, exception=exc)
                _fire(self.environment, "WS", f"{turn_name}_e2e", 0, exception=exc)
                return

            # -- Capture user input transcription (server VAD) --
            if etype == "conversation.item.input_audio_transcription.completed":
                tx = event.get("transcript", "")
                if tx:
                    user_transcript_parts.append(tx)

            # -- Collect assistant transcription fragments --
            if etype == "response.text.delta":
                delta = event.get("delta", "")
                if delta:
                    assistant_text_parts.append(delta)
            elif etype == "response.audio_transcript.delta":
                delta = event.get("delta", "")
                if delta:
                    assistant_text_parts.append(delta)

            if etype in ("response.text.delta", "response.audio.delta", "response.audio_transcript.delta"):
                if t_first_delta is None:
                    t_first_delta = time.perf_counter()
                    ttfb_ms = (t_first_delta - t_audio_done) * 1000
                    _fire(self.environment, "WS", f"{turn_name}_ttfb", ttfb_ms)

            if etype == "response.done":
                resp = event.get("response", {})
                status = resp.get("status", "")

                # If the response was cancelled (e.g. VAD detected more
                # speech), skip it and keep waiting for the real one.
                if status == "cancelled":
                    logger.debug("Turn %d: response cancelled, waiting for next",
                                 turn_idx)
                    # Reset assistant text — partial response is discarded.
                    assistant_text_parts.clear()
                    t_first_delta = None
                    continue

                e2e_ms = (time.perf_counter() - t_audio_done) * 1000
                _fire(self.environment, "WS", f"{turn_name}_e2e", e2e_ms)

                # Extract transcripts from response.done output items
                # only if no streaming deltas were received (fallback).
                if not assistant_text_parts:
                    for item in resp.get("output", []):
                        for content in item.get("content", []):
                            ctype = content.get("type", "")
                            if ctype == "text" and content.get("text"):
                                assistant_text_parts.append(content["text"])
                            elif ctype == "audio" and content.get("transcript"):
                                assistant_text_parts.append(content["transcript"])

                # -- Log transcriptions --
                if transcript_logger:
                    user_text = " ".join(user_transcript_parts).strip()
                    if user_text:
                        transcript_logger.info(
                            "[convo=%s] turn_%d USER  : %s",
                            convo_id, turn_idx, user_text,
                        )
                    else:
                        transcript_logger.info(
                            "[convo=%s] turn_%d USER  : [audio: %s]",
                            convo_id, turn_idx, audio_name,
                        )
                    assistant_text = "".join(assistant_text_parts)
                    if assistant_text:
                        transcript_logger.info(
                            "[convo=%s] turn_%d ASSISTANT: %s",
                            convo_id, turn_idx, assistant_text,
                        )
                return

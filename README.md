# Voice Live Stress Test Tool

A [Locust](https://locust.io/)-based stress test tool for the **Azure AI Speech Voice Live API**.  Each simulated user opens a WebSocket, configures a session, streams a sequence of prerecorded audio files as separate "user turns", and records latency metrics.

## Prerequisites

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Runtime |
| ffmpeg | Required by `pydub` for MP3 → PCM conversion |
| pip | Package manager |

Install ffmpeg (Ubuntu/Debian):

```bash
sudo apt-get install ffmpeg
```

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare audio files

Place your MP3 files (4–5 recommended) in the `./audio/` directory, then convert them to raw PCM16:

```bash
mkdir -p audio
# Copy your .mp3 files into ./audio/

python scripts/audio_preprocess.py --input-dir ./audio --output-dir ./audio_pcm24k --rate 24000
```

This produces `./audio_pcm24k/*.raw` files (mono, 16-bit LE, 24 kHz).

### 3. Configure environment

```bash
export VOICELIVE_ENDPOINT="wss://<resource>.services.ai.azure.com/voice-live/realtime?api-version=2025-10-01&model=<model>"
export VOICELIVE_BEARER_TOKEN="<your-bearer-token>"    # preferred auth
# OR
export VOICELIVE_API_KEY="<your-api-key>"               # alternative auth

export AUDIO_DIR="./audio_pcm24k"
```

### 4. Run Locust

```bash
# Web UI (http://localhost:8089)
locust -f locustfile.py --host dummy

# Headless — 10 users, ramp 2/s, run 2 minutes
locust -f locustfile.py --host dummy --headless -u 10 -r 2 -t 2m
```

> `--host` is required by Locust but unused; set to any value.

## Configuration Reference

All settings are read from environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `VOICELIVE_ENDPOINT` | *(required)* | Full `wss://` URL including `api-version` and `model` params |
| `VOICELIVE_BEARER_TOKEN` | *(empty)* | OAuth bearer token (scope `https://ai.azure.com/.default`) |
| `VOICELIVE_API_KEY` | *(empty)* | API key (appended as query parameter) |
| `AUDIO_DIR` | `./audio_pcm24k` | Folder with `.raw` PCM16 files |
| `TURNS_PER_CONVO` | `4` | Number of audio turns per conversation |
| `CHUNK_MS` | `20` | Chunk duration in milliseconds (20, 50, 100) |
| `STREAM_SPEED` | `1.0` | `1.0` = real-time, `2.0` = 2× speed, `0` = no sleep |
| `SAMPLE_RATE` | `24000` | PCM sample rate (16000 or 24000) |
| `RESPONSE_MODALITIES` | `text,audio` | Comma-separated modalities for responses |
| `VOICE` | *(empty)* | Voice name (e.g., `alloy`) |
| `TURN_DETECTION_TYPE` | *(empty)* | Turn detection type (e.g., `azure_semantic_vad`) |
| `TURN_THINK_TIME_MS` | `0` | Pause between turns in ms |
| `WS_CONNECT_TIMEOUT_S` | `10` | WebSocket connect timeout |
| `TURN_TIMEOUT_S` | `30` | Max wait for `response.done` per turn |

## Metrics

The tool emits these custom Locust metrics (visible in the Web UI and CSV output):

| Metric Name | Type | Description |
|-------------|------|-------------|
| `connect` | WS | WebSocket connection latency |
| `turn_N_ttfb` | WS | Time from commit to first `response.text.delta` or `response.audio.delta` |
| `turn_N_e2e` | WS | Time from commit to `response.done` |
| `conversation_total` | WS | End-to-end conversation time |

Errors are categorized automatically (timeout, server error, WS close, etc.).

## Project Structure

```
├── locustfile.py                # Locust User class and task definition
├── audio.py                     # PCM audio chunking and base64 encoding
├── ws_client.py                 # WebSocket client for Voice Live protocol
├── requirements.txt             # Python dependencies
├── scripts/
│   └── audio_preprocess.py      # MP3 → raw PCM16 conversion script
├── tests/
│   ├── test_audio.py            # Unit tests for audio module
│   └── test_ws_client.py        # Unit tests for WebSocket client
└── README.md
```

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Troubleshooting

### Authentication failures
- **Bearer token**: Ensure the token is valid and has scope `https://ai.azure.com/.default`. Tokens expire — refresh before long runs.
- **API key**: Verify the key matches the resource. It is appended as a query parameter `api-key=...`.

### Model or region mismatch
- The `model` query parameter must match a deployed model in your Azure AI resource.
- Ensure the resource hostname is correct (`.services.ai.azure.com` or `.cognitiveservices.azure.com`).

### Audio format mismatch
- Voice Live expects PCM16 at the configured sample rate (default 24 kHz).
- Use `scripts/audio_preprocess.py` to convert MP3 files.
- Verify with: `ffprobe -i file.raw -f s16le -ar 24000 -ac 1`.

### Timeouts
- Increase `WS_CONNECT_TIMEOUT_S` for slow networks.
- Increase `TURN_TIMEOUT_S` if the model takes long to respond.
- Check that `STREAM_SPEED` is not `0` for large files — this sends all audio instantly and may overwhelm the server.

### No PCM files found
- Ensure `AUDIO_DIR` points to a directory containing `.raw` files.
- Run `scripts/audio_preprocess.py` first to generate them from MP3s.

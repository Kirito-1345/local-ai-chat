# Local AI Chat – LM Studio + Gradio

Gradio-based chat frontend with multimodal support, communicating via LM Studio's OpenAI-compatible REST API. Supports streaming, full conversation history, Base64-encoded image uploads, and local speech-to-text via Whisper.

---

## Stack

- **Frontend/Backend:** [Gradio](https://gradio.app/) `ChatInterface` with `MultimodalTextbox`
- **API Client:** `openai` Python SDK against LM Studio's local OpenAI-compatible endpoint
- **Image handling:** Base64 encoding in-process, no external service
- **Audio handling:** [OpenAI Whisper](https://github.com/openai/whisper) for local speech-to-text transcription

---

## Requirements

- Python 3.9+
- [LM Studio](https://lmstudio.ai/) with Local Server running on `localhost:1234`
- For image analysis: a model with vision support (e.g. LLaVA, Gemma Multimodal)
- For audio/microphone input: `openai-whisper` + `ffmpeg` installed on your system

---

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Gradio binds to `0.0.0.0:7860`, LM Studio is expected at `http://localhost:1234/v1`.

On first launch with audio enabled, Whisper will automatically download the `base` model (~140MB).

---

## Configuration

Overridable via environment variables at runtime:

| Variable | Default | Description |
|---|---|---|
| `LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` | OpenAI-compatible endpoint |
| `LMSTUDIO_MODEL` | `gemma-4-e4b` | Model ID as shown in LM Studio |

```bash
LMSTUDIO_BASE_URL=http://192.168.1.10:1234/v1 LMSTUDIO_MODEL=llava-1.6 python app.py
```

`api_key` is set to `"not-needed"` – LM Studio requires no authentication.

---

## Image Formats

Uploads are Base64-encoded and passed to the API as `image_url` content blocks.

Supported: `jpg`, `jpeg`, `png`, `gif`, `bmp`

WebP is explicitly rejected – LM Studio's vision backend has known issues with the format.

---

## Audio / Microphone

Audio input is transcribed locally via Whisper (`base` model by default) and sent as text to LM Studio. No audio data leaves your machine.

Supported formats: `mp3`, `wav`, `m4a`, `ogg`, `flac`, `webm`

> **Note:** ffmpeg must be installed and available in your PATH for Whisper to work.
> - Windows: `winget install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/)
> - Linux: `sudo apt install ffmpeg`
> - macOS: `brew install ffmpeg`

---

# made during my intern at:

![ScaDS.AI Logo](https://scads.ai/wp-content/themes/scads2023/assets/images/logo.png)

[Center For Scalable Data Analytics And AI](https://scads.ai)

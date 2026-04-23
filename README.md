# Local AI Chat – LM Studio + Gradio

Gradio-based chat frontend with multimodal support, communicating via LM Studio's OpenAI-compatible REST API. Supports streaming, full conversation history, and Base64-encoded image uploads natively.

---

## Stack

- **Frontend/Backend:** [Gradio](https://gradio.app/) `ChatInterface` with `MultimodalTextbox`
- **API Client:** `openai` Python SDK against LM Studio's local OpenAI-compatible endpoint
- **Image handling:** Base64 encoding in-process, no external service

---

## Requirements

- Python 3.9+
- [LM Studio](https://lmstudio.ai/) with Local Server running on `localhost:1234`
- For image analysis: a model with vision support (e.g. LLaVA, Gemma Multimodal)

---

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Gradio binds to `0.0.0.0:7860`, LM Studio is expected at `http://localhost:1234/v1`.

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

## Project Structure

```
.
├── app.py               # Full application logic
├── requirements.txt
└── README.md
```

---

## Dependencies

```
gradio
openai
```

---

## License

MIT
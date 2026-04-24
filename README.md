# Local AI Chat – LiteRT-LM + Gradio

Gradio-based chat frontend with multimodal support, communicating with a local LiteRT-LM runtime via an OpenAI-compatible REST API.

Supports streaming-style responses, full conversation history, Base64 image uploads, and **direct audio input handled by Gemma 4 (no Whisper required)**.

---

## Stack

* **Frontend:** [Gradio](https://gradio.app/) (`ChatInterface`, `MultimodalTextbox`)
* **API Client:** [`openai`](https://pypi.org/project/openai/) Python SDK (against local endpoint)
* **Model Runtime:** [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)
* **API Layer:** [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
* **Multimodal:** Text, image, and audio handled directly by Gemma 4

---

## Requirements

* Python 3.12
* [uv](https://docs.astral.sh/uv/)
* Hugging Face account
* Hugging Face access token
* Access to Gemma 4 (gated model)

Recommended for local usage:

```text
gemma-4-E4B-it.litertlm
or
gemma-4-E2B-it.litertlm
```

> E4B may also work but is heavier and slower on CPU/iGPU systems.
Check this out and use these models: [Link](https://huggingface.co/litert-community)

---

## Setup

git clone...
```bash
git clone <repo>
cd ...
```

Install dependencies:

```bash
uv sync
```

## Hugging Face Setup (Important)

Gemma models are gated on Hugging Face. You must:

1. Log into Hugging Face in your browser
2. Open the Gemma 4 model page
3. Accept the model terms
4. Create an access token
5. Authenticate locally

### Huggingface-Hub

Maybe you need granted access to certain models like gemma4; therefore you need an HF account and need to accept terms of service (ToS) of e.g. "gated" models.
Afterwards you possibly need a HF access token to donwload it via your account and your agreement of ToS.
The respective package for HF-Hub is already in pyproject.toml; if necessary  install HF-CLI:

```bash
uv add "huggingface_hub[cli]"
```

### Check login

```bash
uv run hf auth whoami
```

### Login (if needed)

```bash
uv run hf auth login
```

Paste your access token. 
You can create tokens with your huggingface-account, see [this](https://huggingface.co/docs/hub/security-tokens).

### Set local cache directory (where e.g. HF models are stored)

```bash
export HF_HOME="./huggingface-hub"
```

Windows PowerShell:

```powershell
$env:HF_HOME = "./huggingface-hub"
```

---

---

## Install & Test LiteRT-LM 

Install CLI:

```bash
uv tool install litert-lm
```

Run a test:

```bash
litert-lm run \
  --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm \
  gemma-4-E2B-it.litertlm \
  --prompt="Hello, who are you?"
```

If this works, your setup is correct.

---

## Start LiteRT-LM API Server

```bash
export HF_HOME="./huggingface-hub"
export LITERT_MODEL_PATH="./huggingface-hub/hub/models--litert-community--gemma-4-E4B-it-litert-lm/snapshots/abcdefg12345.../gemma-4-E4B-it.litertlm
export LITERT_BACKEND="CPU"

uv run uvicorn litert_server:app \
  --host 0.0.0.0 \
  --port 8000
```

---

## Start Gradio App

```bash
export LOCAL_LLM_BASE_URL="http://localhost:8000/v1"
export LOCAL_LLM_MODEL="gemma-4-E4B-it"

uv run python app.py
```

Open:

```text
http://localhost:7860
```

---

## Configuration

| Variable             | Default                     | Description                   |
| -------------------- | --------------------------- | ----------------------------- |
| `LOCAL_LLM_BASE_URL` | `http://localhost:8000/v1`  | Local API endpoint            |
| `LOCAL_LLM_MODEL`    | `gemma-4-E4B-it`            | Model name                    |
| `LOCAL_LLM_API_KEY`  | `EMPTY`                     | Dummy value                   |
| `LITERT_MODEL_PATH`  | `./gemma-4-E2B-it.litertlm` | Model file path               |
| `LITERT_BACKEND`     | `CPU`                       | Backend (`CPU`, `GPU`, `NPU`) |
| `HF_HOME`            | system default              | Hugging Face cache            |

---

## Image Formats

Supported:

```text
jpg, jpeg, png, gif, bmp
```

WebP is intentionally blocked to avoid compatibility issues.

---

## Audio / Microphone

* No Whisper required ❌
* Audio is sent directly to Gemma 4 ✅

Supported:

```text
mp3, wav, m4a, ogg, flac, webm
```

All processing stays local.

---

## Project Structure

```text
local-ai-chat/
├── app.py
├── litert_lm_server.py
├── pyproject.toml
├── README.md
└── huggingface-hub/ (optional)
```

---

## Development Workflow

```bash
uv sync
uv run hf auth whoami  # check login
uv run hf auth login   # if needed

export HF_HOME="./huggingface-hub"

# start backend
uv run uvicorn litert_server:app --host 0.0.0.0 --port 8000

# start frontend
uv run python app.py
```

---

# made during my intern at:

![ScaDS.AI Logo](https://scads.ai/wp-content/themes/scads2023/assets/images/logo.png)

[Center For Scalable Data Analytics And AI](https://scads.ai)


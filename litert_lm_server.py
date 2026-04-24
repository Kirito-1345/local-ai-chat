import base64
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import litert_lm
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


MODEL_PATH = os.getenv("LITERT_MODEL_PATH", "./gemma-4-E2B-it.litertlm")
LITERT_BACKEND = os.getenv("LITERT_BACKEND", "CPU").upper()

app = FastAPI(title="LiteRT-LM OpenAI-compatible local API")
engine = None


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None


def pick_backend(name: str):
    if name == "GPU":
        return litert_lm.Backend.GPU
    if name == "NPU":
        return litert_lm.Backend.NPU
    return litert_lm.Backend.CPU


def suffix_from_data_url(data_url: str, fallback: str) -> str:
    header = data_url.split(",", 1)[0].lower()

    if "image/jpeg" in header:
        return ".jpg"
    if "image/png" in header:
        return ".png"
    if "image/gif" in header:
        return ".gif"
    if "image/bmp" in header:
        return ".bmp"
    if "audio/wav" in header or "audio/x-wav" in header:
        return ".wav"
    if "audio/mpeg" in header or "audio/mp3" in header:
        return ".mp3"
    if "audio/mp4" in header:
        return ".m4a"
    if "audio/ogg" in header:
        return ".ogg"
    if "audio/flac" in header:
        return ".flac"
    if "audio/webm" in header:
        return ".webm"

    return fallback


def data_url_to_temp_file(data_url: str, fallback_suffix: str) -> str:
    _, b64 = data_url.split(",", 1)
    raw = base64.b64decode(b64)
    suffix = suffix_from_data_url(data_url, fallback_suffix)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(raw)
    tmp.close()

    return tmp.name


def openai_part_to_litert(part: dict[str, Any]) -> dict[str, Any]:
    part_type = part.get("type")

    if part_type == "text":
        return {
            "type": "text",
            "text": part.get("text", ""),
        }

    if part_type == "image_url":
        url = part.get("image_url", {}).get("url", "")

        if url.startswith("data:"):
            return {
                "type": "image",
                "path": data_url_to_temp_file(url, ".jpg"),
            }

        return {
            "type": "image",
            "path": url,
        }

    if part_type == "audio_url":
        url = part.get("audio_url", {}).get("url", "")

        if url.startswith("data:"):
            return {
                "type": "audio",
                "path": data_url_to_temp_file(url, ".wav"),
            }

        return {
            "type": "audio",
            "path": url,
        }

    raise ValueError(f"Unsupported content part type: {part_type}")


def openai_message_to_litert(message: dict[str, Any]) -> dict[str, Any]:
    role = message.get("role", "user")
    content = message.get("content", "")

    if isinstance(content, str):
        return {
            "role": role,
            "content": content,
        }

    if isinstance(content, list):
        return {
            "role": role,
            "content": [openai_part_to_litert(part) for part in content],
        }

    return {
        "role": role,
        "content": str(content),
    }


def extract_text(response: Any) -> str:
    """
    LiteRT-LM Responses können je nach Version leicht variieren.
    Diese Funktion ist absichtlich defensiv.
    """
    if isinstance(response, str):
        return response

    if isinstance(response, dict):
        content = response.get("content", response)

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    chunks.append(item.get("text", ""))
                elif isinstance(item, str):
                    chunks.append(item)
            return "".join(chunks)

        if "text" in response:
            return str(response["text"])

    if hasattr(response, "text"):
        return str(response.text)

    return str(response)


@app.on_event("startup")
def startup():
    global engine

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"LiteRT-LM model not found: {MODEL_PATH}. "
            "Set LITERT_MODEL_PATH or place the .litertlm file there."
        )

    litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

    backend = pick_backend(LITERT_BACKEND)

    # CPU als Default ist auf deiner Maschine am wahrscheinlichsten robust.
    # Wenn LiteRT-LM GPU auf deinem Gerät unterstützt, kannst du LITERT_BACKEND=GPU setzen.
    engine = litert_lm.Engine(
        MODEL_PATH,
        backend=backend,
        vision_backend=backend,
        audio_backend=litert_lm.Backend.CPU,
    )

    print(f"LiteRT-LM loaded: {MODEL_PATH}")
    print(f"Backend: {LITERT_BACKEND}")


@app.on_event("shutdown")
def shutdown():
    global engine

    if engine is not None:
        engine.close()
        engine = None


@app.get("/v1/models")
def list_models():
    model_name = Path(MODEL_PATH).stem

    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if engine is None:
        raise RuntimeError("LiteRT-LM engine is not initialized.")

    litert_messages = [openai_message_to_litert(m) for m in req.messages]

    # Simpler stateless Betrieb:
    # Jeder Request enthält den Chatverlauf vollständig.
    with engine.create_conversation() as conversation:
        response = None

        for message in litert_messages:
            response = conversation.send_message(message)

    text = extract_text(response)

    if req.stream:
        def event_stream():
            # Minimaler OpenAI-kompatibler Streaming-Fallback:
            # kein echtes Tokenstreaming, aber kompatibel genug für viele Clients.
            chunk = {
                "id": f"chatcmpl-litert-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model or Path(MODEL_PATH).stem,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None,
                    }
                ],
            }
            import json
            yield f"data: {json.dumps(chunk)}\n\n"

            done = {
                "id": f"chatcmpl-litert-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model or Path(MODEL_PATH).stem,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return {
        "id": f"chatcmpl-litert-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or Path(MODEL_PATH).stem,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }
        ],
    }
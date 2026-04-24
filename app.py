import base64
import os
from pathlib import Path
from typing import Any

import gradio as gr
from openai import OpenAI


# 👉 passt jetzt zu deinem LiteRT-Server
DEFAULT_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = os.getenv("LOCAL_LLM_MODEL", "gemma-4-E4B-it")
DEFAULT_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "EMPTY")

client = OpenAI(
    base_url=DEFAULT_BASE_URL,
    api_key=DEFAULT_API_KEY,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}

MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
}


def extract_file_path(file_item: Any) -> str:
    if isinstance(file_item, str):
        return file_item

    if isinstance(file_item, dict):
        return (
            file_item.get("path")
            or file_item.get("name")
            or file_item.get("file", {}).get("path")
            or ""
        )

    return getattr(file_item, "path", None) or getattr(file_item, "name", "") or ""


def file_to_data_url(file_path: str) -> str:
    path = Path(file_path)
    ext = path.suffix.lower()

    mime = MIME_MAP.get(ext)
    if not mime:
        raise ValueError(f"Unsupported file type: {ext}")

    with path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{encoded}"


def content_part_from_file(file_path: str) -> dict:
    ext = Path(file_path).suffix.lower()

    if ext == ".webp":
        raise ValueError("WebP not supported. Use JPG/PNG/GIF/BMP.")

    if ext in IMAGE_EXTENSIONS:
        return {
            "type": "image_url",
            "image_url": {"url": file_to_data_url(file_path)},
        }

    if ext in AUDIO_EXTENSIONS:
        return {
            "type": "audio_url",
            "audio_url": {"url": file_to_data_url(file_path)},
        }

    raise ValueError(f"Unsupported file type: {ext}")


def build_user_content(text: str, files: list[Any]):
    content = []

    for file_item in files or []:
        file_path = extract_file_path(file_item)

        if not file_path:
            continue

        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        content.append(content_part_from_file(file_path))

    if text:
        content.append({"type": "text", "text": text})

    if not content:
        return ""

    if len(content) == 1 and content[0]["type"] == "text":
        return content[0]["text"]

    return content


def normalize_history(history):
    messages = []

    for msg in history or []:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        content = msg.get("content", "")

        if role not in {"user", "assistant", "system"}:
            continue

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            text_parts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]

            if text_parts:
                messages.append(
                    {"role": role, "content": "\n".join(text_parts)}
                )

    return messages


def chat(message, history):
    text = (message or {}).get("text", "") or ""
    files = (message or {}).get("files", []) or []

    if not text and not files:
        yield "⚠️ Please enter a message or upload a file."
        return

    try:
        user_content = build_user_content(text, files)
    except ValueError as e:
        yield f"⚠️ {e}"
        return

    messages = normalize_history(history)
    messages.append({"role": "user", "content": user_content})

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            stream=True,
        )

        partial = ""

        for chunk in response:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                partial += delta.content
                yield partial

    except Exception as e:
        yield (
            "Error connecting to LiteRT-LM server\n\n"
            f"Endpoint: {DEFAULT_BASE_URL}\n"
            f"Model: {DEFAULT_MODEL}\n\n"
            f"{e}"
        )


demo = gr.ChatInterface(
    fn=chat,
    title="Local AI Chat",
    description="LiteRT-LM backend (Gemma 4 multimodal: text + image + audio)",
    multimodal=True,
    textbox=gr.MultimodalTextbox(
        file_count="multiple",
        file_types=["image", "audio"],
        sources=["upload", "microphone"],
        placeholder="Type a message, upload image/audio, or use mic...",
        show_label=False,
    ),
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
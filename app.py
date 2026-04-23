import gradio as gr
from openai import OpenAI
import os
import base64

DEFAULT_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
DEFAULT_MODEL = os.getenv("LMSTUDIO_MODEL", "gemma-4-e4b")

client = OpenAI(
    base_url=DEFAULT_BASE_URL,
    api_key="not-needed"
)

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


def file_to_data_url(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    mime = MIME_MAP.get(ext, "image/jpeg")
    with open(file_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{image_data}"


def build_content(text, files):
    if not files:
        return text or ""

    content = []
    for file_path in files:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".webp":
            raise ValueError("WebP-Bilder werden nicht unterstützt. Bitte konvertiere das Bild zu JPG oder PNG.")
        if ext not in SUPPORTED_FORMATS:
            raise ValueError(f"Das Format '{ext}' wird nicht unterstützt. Bitte verwende JPG, PNG, GIF oder BMP.")
        content.append({
            "type": "image_url",
            "image_url": {"url": file_to_data_url(file_path)}
        })

    if text:
        content.append({"type": "text", "text": text})

    return content


def chat(message, history):
    messages = []

    for msg in history:
        if isinstance(msg, dict):
            role = msg["role"]
            raw = msg["content"]

            if isinstance(raw, list):
                content = []
                for item in raw:
                    if item.get("type") == "file":
                        file_path = item.get("file", {}).get("path", "")
                        if file_path and os.path.exists(file_path):
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": file_to_data_url(file_path)}
                            })
                    elif item.get("type") == "text":
                        content.append({"type": "text", "text": item.get("text", "")})
                messages.append({"role": role, "content": content})
            else:
                messages.append({"role": role, "content": str(raw)})

    text = message.get("text", "") or ""
    files = message.get("files", []) or []

    try:
        user_content = build_content(text, files)
    except ValueError as e:
        yield f"⚠️ {str(e)}"
        return

    messages.append({"role": "user", "content": user_content})

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            stream=True
        )

        partial_message = ""
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content is not None:
                partial_message += delta.content
                yield partial_message

    except Exception as e:
        yield f"Fehler bei der Verbindung zu LM Studio: {str(e)}"


demo = gr.ChatInterface(
    chat,
    title="Lokaler AI Chat",
    description="Verbunden mit LM Studio via OpenAI-kompatibler API",
    multimodal=True,
    textbox=gr.MultimodalTextbox(
        file_count="multiple",
        file_types=["image"],
        sources=["upload"],
        placeholder="Nachricht eingeben oder Bild hochladen... (kein WebP!)",
        show_label=False,
    ),
    examples=[
        {"text": "Hallo, wer bist du?", "files": []},
        {"text": "Erkläre mir Quantenphysik einfach.", "files": []},
        {"text": "Schreibe ein Python-Skript, das eine Liste sortiert.", "files": []},
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
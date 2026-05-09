import requests
from config import OLLAMA_BASE_URL, EMBED_MODEL_NAME, MODEL_NAME


def get_embedding(text: str) -> list[float]:
    """调用 Ollama embedding API 获取文本向量"""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL_NAME, "prompt": text},
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def chat(messages: list[dict], model: str = MODEL_NAME) -> str:
    """调用 Ollama chat API 进行对话"""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": model, "messages": messages, "stream": False},
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]
import json
from collections.abc import Iterator

import requests

from .config import get_settings


class ProviderError(RuntimeError):
    pass


class ChatProvider:
    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        raise NotImplementedError


class MockProvider(ChatProvider):
    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        question = messages[-1]["content"]
        answer = (
            "Smart Q&A is running in mock mode. Configure AI_PROVIDER with "
            "nvidia or openai for model-backed answers. You asked: " + question
        )
        for word in answer.split(" "):
            yield word + " "


class NvidiaProvider(ChatProvider):
    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        settings = get_settings()
        if not settings.nvidia_api_key:
            raise ProviderError("NVIDIA_API_KEY is required")
        response = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.nvidia_api_key}"},
            json={"model": settings.chat_model, "messages": messages, "stream": True, "max_tokens": 1200},
            stream=True,
            timeout=120,
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                return
            event = json.loads(payload)
            text = event["choices"][0].get("delta", {}).get("content", "")
            if text:
                yield text


class OpenAIProvider(ChatProvider):
    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        settings = get_settings()
        if not settings.openai_api_key:
            raise ProviderError("OPENAI_API_KEY is required")
        instructions = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        input_messages = [m for m in messages if m["role"] != "system"]
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json={
                "model": settings.chat_model,
                "instructions": instructions,
                "input": input_messages,
                "stream": True,
            },
            stream=True,
            timeout=120,
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                return
            event = json.loads(payload)
            if event.get("type") == "response.output_text.delta":
                yield event.get("delta", "")


def chat_provider() -> ChatProvider:
    provider = get_settings().ai_provider.lower()
    if provider == "nvidia":
        return NvidiaProvider()
    if provider == "openai":
        return OpenAIProvider()
    return MockProvider()


def embed_texts(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    if not texts:
        return []
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise ProviderError("OPENAI_API_KEY is required for embeddings")
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json={"model": settings.embedding_model, "input": texts},
            timeout=120,
        )
        response.raise_for_status()
        return [item["embedding"] for item in sorted(response.json()["data"], key=lambda row: row["index"])]
    if settings.embedding_provider == "mock":
        return [_mock_embedding(text) for text in texts]
    if not settings.nvidia_api_key:
        raise ProviderError("NVIDIA_API_KEY is required for embeddings")
    response = requests.post(
        "https://integrate.api.nvidia.com/v1/embeddings",
        headers={"Authorization": f"Bearer {settings.nvidia_api_key}"},
        json={
            "model": settings.embedding_model,
            "input": texts,
            "input_type": "passage",
            "encoding_format": "float",
            "truncate": "END",
        },
        timeout=120,
    )
    response.raise_for_status()
    return [item["embedding"] for item in sorted(response.json()["data"], key=lambda row: row["index"])]


def embed_query(text: str) -> list[float]:
    settings = get_settings()
    if settings.embedding_provider == "nvidia" and settings.nvidia_api_key:
        response = requests.post(
            "https://integrate.api.nvidia.com/v1/embeddings",
            headers={"Authorization": f"Bearer {settings.nvidia_api_key}"},
            json={
                "model": settings.embedding_model,
                "input": [text],
                "input_type": "query",
                "encoding_format": "float",
                "truncate": "END",
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    return embed_texts([text])[0]


def _mock_embedding(text: str) -> list[float]:
    values = [0.0] * 32
    for index, byte in enumerate(text.encode("utf-8")):
        values[index % len(values)] += byte / 255
    magnitude = sum(value * value for value in values) ** 0.5 or 1
    return [value / magnitude for value in values]

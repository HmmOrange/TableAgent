from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import requests

from configs.config import load_config
from configs.embedding_config import resolve_embedding_config


class MockEmbeddingModel:
    """A deterministic embedding model for offline retrieval tests."""

    def __init__(self, dim: int = 128):
        self.dim = dim

    async def encode(self, texts: str | list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            words = re.findall(r"\w+", text.lower()) or [""]
            for word in words:
                index = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self.dim
                vec[index] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)
        return np.array(embeddings)

    async def batch_encode(self, texts: str | list[str], **kwargs) -> np.ndarray:
        return await self.encode(texts)


class OpenAICompatibleEmbeddingClient:
    """Minimal OpenAI-compatible embedding client used by retrieval backends."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float | None = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "EMPTY"
        self.timeout = timeout

    @classmethod
    def from_config(cls, provider: str | None = None, config_path: str | Path = "configs/config.yaml"):
        config = load_config(config_path)
        provider_name, embedding_config = resolve_embedding_config(config, provider or "embedding")
        backend = str(embedding_config.get("provider", "")).lower()
        if backend not in {"openai", "openai_compatible"}:
            raise ValueError(
                f"Unsupported retrieval embedding backend for '{provider_name}': {backend}. "
                "Only OpenAI-compatible embeddings are supported."
            )
        base_url = embedding_config.get("base_url") or embedding_config.get("endpoint")
        model = embedding_config.get("model") or embedding_config.get("model_name")
        if not base_url or not model:
            raise ValueError(
                f"Embedding provider '{provider_name}' must configure base_url/endpoint and model/model_name."
            )
        timeout = embedding_config.get("timeout", embedding_config.get("timeout_seconds", 120))
        return cls(
            base_url=str(base_url),
            model=str(model),
            api_key=embedding_config.get("api_key"),
            timeout=float(timeout) if timeout is not None else None,
        )

    async def encode(self, texts: str | list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        response.raise_for_status()
        rows = sorted(response.json().get("data", []), key=lambda row: row.get("index", 0))
        return np.array([row["embedding"] for row in rows], dtype=np.float32)

    async def batch_encode(self, texts: str | list[str], **kwargs) -> np.ndarray:
        return await self.encode(texts)
